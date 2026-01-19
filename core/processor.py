import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import cv2
import numpy as np
import insightface
import torch
import time
import onnx
import threading
import queue
import subprocess
from functools import lru_cache
from core.checkpoint_manager import CheckpointManager, log_with_time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union, ContextManager
from onnxruntime import InferenceSession
from core.config import ModelManager, get_face_model
from core.auto_pixel_boost import AutoPixelBoostSelector, get_recommended_pixel_boost

# 自动安装psutil
try:
    import psutil
except ImportError:
    print("[INFO] 检测到缺少psutil库，正在自动安装...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "-q"])
        import psutil
        print("[INFO] psutil安装成功")
    except Exception as e:
        print(f"[WARNING] psutil安装失败: {e}，将使用保守的内存估算")
        psutil = None

# ===== GPU显存平衡器 =====
class GPUMemoryBalancer:
    """GPU显存平衡器 - 自动处理GPU0的额外开销,智能分配worker数量"""
    
    def __init__(self, gpu_ids, max_workers_per_gpu=4, debug=False):
        self.gpu_ids = gpu_ids
        self.max_workers_per_gpu = max_workers_per_gpu
        self.debug = debug
        
        # 预估参数(可根据实际模型调整)
        self.MODEL_BASE_MEMORY = 0.7     # 模型基础显存(GB)
        self.WORKER_MEMORY = 1.6         # 每个worker额外显存(GB)
        self.GPU0_OVERHEAD = 1.9         # GPU0额外开销(GB)
        self.SAFETY_MARGIN = 0.88        # 安全系数
        
    def get_gpu_memory_info(self):
        """获取所有GPU的显存信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            gpu_info = {}
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gid = int(parts[0])
                        gpu_info[gid] = {
                            'total': float(parts[1]) / 1024,
                            'used': float(parts[2]) / 1024,
                            'free': float(parts[3]) / 1024
                        }
            return gpu_info
        except Exception as e:
            log_with_time("WARNING", f"无法获取GPU显存信息: {e}")
            return {}
    
    def calculate_workers_per_gpu(self):
        """计算每个GPU的最优worker数量，考虑GPU0的额外开销"""
        gpu_info = self.get_gpu_memory_info()
        workers_config = {}
        
        if not gpu_info:
            log_with_time("WARNING", "无法获取GPU显存,使用保守分配策略")
            for gpu_id in self.gpu_ids:
                workers_config[gpu_id] = min(2 if gpu_id == 0 else 3, self.max_workers_per_gpu)
            return workers_config
        
        # 一轮分配：GPU0扣除额外开销后，与其他GPU使用相同计算方式
        for gpu_id in self.gpu_ids:
            if gpu_id not in gpu_info:
                workers_config[gpu_id] = 2
                continue
            
            info = gpu_info[gpu_id]
            free_memory = info['free']
            
            # GPU0需要预留额外开销
            if gpu_id == 0:
                effective_memory = (free_memory - self.GPU0_OVERHEAD) * self.SAFETY_MARGIN
            else:
                effective_memory = free_memory * self.SAFETY_MARGIN
            
            # 计算worker数量
            workers = int((effective_memory - self.MODEL_BASE_MEMORY) / self.WORKER_MEMORY)
            workers = max(1, min(workers, self.max_workers_per_gpu))
            
            workers_config[gpu_id] = workers
        
        return workers_config
    
    def print_allocation_summary(self, workers_config, gpu_info):
        """打印分配摘要"""
        log_with_time("INFO", "\n" + "="*60)
        log_with_time("INFO", "GPU Worker 分配方案")
        log_with_time("INFO", "="*60)
        
        total_workers = 0
        for gpu_id in sorted(self.gpu_ids):
            num_workers = workers_config.get(gpu_id, 0)
            total_workers += num_workers
            
            # 计算预估使用
            estimated = self.MODEL_BASE_MEMORY + num_workers * self.WORKER_MEMORY
            if gpu_id == 0:
                estimated += self.GPU0_OVERHEAD
            
            # 显示信息
            if gpu_id in gpu_info:
                info = gpu_info[gpu_id]
                usage_pct = (estimated / info['total']) * 100 if info['total'] > 0 else 0
                
                log_with_time("INFO", 
                    f"GPU {gpu_id}: {num_workers} workers | "
                    f"显存 {info['used']:.1f}/{info['total']:.1f}GB | "
                    f"预估使用 {estimated:.1f}GB ({usage_pct:.0f}%)"
                )
                
                if gpu_id == 0:
                    log_with_time("INFO", 
                        f"  └─ 包含: 基础{self.MODEL_BASE_MEMORY:.1f}GB + "
                        f"worker {num_workers}×{self.WORKER_MEMORY:.1f}GB + "
                        f"开销{self.GPU0_OVERHEAD:.1f}GB")
            else:
                log_with_time("INFO", f"GPU {gpu_id}: {num_workers} workers (显存信息未知)")
        
        log_with_time("INFO", f"总计: {total_workers} workers")
        log_with_time("INFO", "="*60 + "\n")
    
    def balance_and_allocate(self):
        """执行完整的平衡和分配流程"""
        log_with_time("INFO", f"开始GPU显存平衡计算 (最大{self.max_workers_per_gpu} workers/GPU)")
        
        gpu_info = self.get_gpu_memory_info()
        workers_config = self.calculate_workers_per_gpu()
        self.print_allocation_summary(workers_config, gpu_info)
        
        return workers_config

# ===== Segment Writer =====
class SegmentWriter:
    """带验证的segment写入器 - 使用 ffmpeg pipe 直接写入 H.264"""
    def __init__(self, checkpoint, segment_idx, video_info):
        self.checkpoint = checkpoint
        self.segment_idx = segment_idx
        self.video_info = video_info
        self.segment_path = checkpoint.get_segment_path(segment_idx)
        
        encoder = video_info.get('encoder', 'NONE')
        log_with_time("INFO", f"Segment {segment_idx} 使用编码器: {encoder}")
        
        self.frames_written = 0
        self.expected_frame_indices = []
        self.actual_frame_indices = []
        self.lock = threading.Lock()
        
        self.ffmpeg_process = None
        self._start_ffmpeg_pipe()

    def _get_fps_rational(self, fps):
        """将浮点数帧率转换为精确的分数表示
        
        Args:
            fps: 浮点数帧率（如 29.97, 23.976）
        
        Returns:
            字符串形式的分数（如 "30000/1001", "24000/1001"）
        """
        from fractions import Fraction
        
        # 常见的帧率映射
        common_fps = {
            23.976: "24000/1001",
            24.0: "24/1",
            25.0: "25/1",
            29.97: "30000/1001",
            30.0: "30/1",
            50.0: "50/1",
            59.94: "60000/1001",
            60.0: "60/1",
        }
        
        # 检查是否是常见帧率（允许0.01的误差）
        for known_fps, rational in common_fps.items():
            if abs(fps - known_fps) < 0.01:
                return rational
        
        # 对于非标准帧率，使用 Fraction 自动转换
        try:
            # 限制分母最大值，避免产生过大的数字
            frac = Fraction(fps).limit_denominator(10000)
            rational = f"{frac.numerator}/{frac.denominator}"
            return rational
        except:
            # 降级：直接使用浮点数（可能不精确）
            log_with_time("WARNING", f"无法转换帧率 {fps}，使用浮点数表示")
            return str(fps)

    def _start_ffmpeg_pipe(self):
        """启动 ffmpeg pipe 进程 - 修复版（显式设置timebase）"""
        width = self.video_info['width']
        height = self.video_info['height']
        fps = self.video_info['fps']
        
        # 获取编码器配置
        encoder = self.video_info.get('encoder', 'libx264')
        crf = self.video_info.get('crf', 23)
        preset = self.video_info.get('preset', 'medium')
        
        # 测试编码器是否可用
        test_cmd = ['ffmpeg', '-hide_banner', '-encoders']
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
            if encoder not in result.stdout:
                log_with_time("WARNING", f"编码器 {encoder} 不可用，回退到 libx264")
                encoder = 'libx264'
                crf = 23
                preset = 'medium'
        except Exception as e:
            log_with_time("WARNING", f"无法检测编码器: {e}，使用默认 libx264")
            encoder = 'libx264'
        
        # 计算精确的帧率分数
        fps_rational = self._get_fps_rational(fps)
        
        # ffmpeg 命令：从 stdin 读取原始帧
        cmd = [
            'ffmpeg', '-y', 
            '-f', 'rawvideo', 
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', 
            '-pix_fmt', 'bgr24', 
            '-r', fps_rational,  # 使用精确的帧率分数
            '-i', '-', 
            '-c:v', encoder,
        ]
        
        # 根据编码器类型添加不同的参数
        if 'nvenc' in encoder:
            cmd.extend(['-preset', preset, '-cq', str(crf), '-b:v', '0', '-rc', 'vbr'])
        elif encoder == 'h264_qsv':
            cmd.extend(['-preset', preset, '-global_quality', str(crf)])
        elif encoder == 'h264_videotoolbox':
            cmd.extend(['-b:v', f'{int(5000 * (51 - crf) / 28)}k'])
        else:  # libx264
            cmd.extend(['-crf', str(crf), '-preset', preset])
        
        # ===== 关键修复：显式设置 timebase =====
        cmd.extend([
            '-pix_fmt', 'yuv420p',
            '-video_track_timescale', '10240',  # 强制使用标准的 timebase
            '-movflags', '+faststart',
            self.segment_path
        ])
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=10**8
            )
            
            time.sleep(0.1)
            if self.ffmpeg_process.poll() is not None:
                stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                log_with_time("ERROR", f"ffmpeg启动失败: {stderr[:500]}")
                self.ffmpeg_process = None
        except Exception as e:
            log_with_time("ERROR", f"ffmpeg pipe 启动失败: {e}")
            self.ffmpeg_process = None

    def write(self, frame, frame_idx):
        """写入帧"""
        with self.lock:
            if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                log_with_time("ERROR", f"Segment {self.segment_idx} ffmpeg进程异常")
                return False
            
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.frames_written += 1
                self.actual_frame_indices.append(frame_idx)
                return True
            except Exception as e:
                log_with_time("ERROR", f"Segment {self.segment_idx} 写入帧失败: {e}")
                return False
    
    def expect_frame(self, frame_idx):
        """记录期望的帧"""
        with self.lock:
            self.expected_frame_indices.append(frame_idx)
    
    def release(self):
        """释放并验证 - 修复超时问题"""
        with self.lock:
            if self.ffmpeg_process is not None:
                try:
                    self.ffmpeg_process.stdin.close()
                    
                    # 根据帧数动态调整超时时间
                    base_timeout = 30
                    extra_timeout = (self.frames_written // 100) * 10
                    total_timeout = min(base_timeout + extra_timeout, 300)
                    
                    log_with_time("INFO", f"Segment {self.segment_idx} 等待ffmpeg完成 ({self.frames_written}帧, 超时{total_timeout}秒)")
                    
                    try:
                        self.ffmpeg_process.wait(timeout=total_timeout)
                        
                        if self.ffmpeg_process.returncode != 0:
                            stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                            log_with_time("ERROR", 
                                f"Segment {self.segment_idx} ffmpeg 编码失败 (返回码 {self.ffmpeg_process.returncode}): {stderr[:200]}")
                        else:
                            log_with_time("INFO", f"Segment {self.segment_idx} ffmpeg 编码完成")
                        
                    except subprocess.TimeoutExpired:
                        log_with_time("ERROR", f"Segment {self.segment_idx} ffmpeg 超时 ({total_timeout}秒)")
                        self.ffmpeg_process.kill()
                        self.ffmpeg_process.wait(timeout=5)
                        
                except Exception as e:
                    log_with_time("ERROR", f"Segment {self.segment_idx} 关闭异常: {e}")
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass
                
                self.ffmpeg_process = None
            
            # 验证：期望的帧和实际写入的帧是否一致
            if self.expected_frame_indices != self.actual_frame_indices:
                missing = set(self.expected_frame_indices) - set(self.actual_frame_indices)
                extra = set(self.actual_frame_indices) - set(self.expected_frame_indices)
                
                if missing:
                    log_with_time("ERROR", 
                        f"Segment {self.segment_idx} 缺失帧: {sorted(missing)[:10]}...")
                if extra:
                    log_with_time("WARNING", 
                        f"Segment {self.segment_idx} 额外帧: {sorted(extra)[:10]}...")
            
            return self.frames_written
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'written': self.frames_written,
                'expected': len(self.expected_frame_indices),
                'actual': len(self.actual_frame_indices)
            }

# ===== 工具函数 =====
def verify_segment_file(segment_path, expected_frames, debug=False):
    """验证segment文件的完整性"""
    if not os.path.exists(segment_path):
        log_with_time("ERROR", f"Segment文件不存在: {segment_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(segment_path)
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 逐帧验证（更准确但慢）
        if debug or abs(reported_frames - expected_frames) > 1:
            actual_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                actual_frames += 1
            
            cap.release()
            
            if actual_frames != expected_frames:
                log_with_time("WARNING", 
                    f"Segment帧数不匹配: 期望 {expected_frames}, "
                    f"实际 {actual_frames}, 报告 {reported_frames}")
                return False
            return True
        else:
            cap.release()
            
            if abs(reported_frames - expected_frames) > 1:
                log_with_time("WARNING", 
                    f"Segment帧数可能不准: 期望 {expected_frames}, "
                    f"报告 {reported_frames}")
                return False
            return True
    
    except Exception as e:
        log_with_time("ERROR", f"验证segment失败: {e}")
        return False

def check_gpu_memory_available(gpu_id, required_gb=2.0):
    """检查GPU是否有足够可用显存"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0] == str(gpu_id):
                    free_mb = float(parts[1])
                    free_gb = free_mb / 1024
                    return free_gb >= required_gb, free_gb
        return False, 0
    except Exception as e:
        log_with_time("WARNING", f"无法检查GPU{gpu_id}显存: {e}")
        return True, 0

def cleanup_gpu_memory(gpu_id):
    """尝试清理GPU显存碎片"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except:
        pass

# 线程同步
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
NULL_CONTEXT = nullcontext()

def thread_semaphore():
    return THREAD_SEMAPHORE

def conditional_thread_semaphore():
    return THREAD_SEMAPHORE

def cosine_sim(a, b):
    """计算余弦相似度"""
    if a is None or b is None:
        return -1.0
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

# ===== 图像处理工具 =====
def unpack_resolution(resolution_str: str) -> Tuple[int, int]:
    """解析分辨率字符串 '512x512' -> (512, 512)"""
    if 'x' in resolution_str:
        width, height = map(int, resolution_str.split('x'))
        return width, height
    return 256, 256

def normalize_resolution(resolution: Tuple[float, float]) -> Tuple[int, int]:
    """标准化分辨率为偶数"""
    width, height = resolution
    if width > 0 and height > 0:
        normalize_width = round(width / 2) * 2
        normalize_height = round(height / 2) * 2
        return normalize_width, normalize_height
    return 0, 0

def implode_pixel_boost(crop_vision_frame, pixel_boost_total: int, model_size: Tuple[int, int]):
    """分解高分辨率图像为模型大小的块"""
    pixel_boost_vision_frame = crop_vision_frame.reshape(
        model_size[1], pixel_boost_total, 
        model_size[0], pixel_boost_total, 3
    )
    pixel_boost_vision_frame = pixel_boost_vision_frame.transpose(1, 3, 0, 2, 4).reshape(
        pixel_boost_total ** 2, model_size[1], model_size[0], 3
    )
    return pixel_boost_vision_frame

def explode_pixel_boost(temp_vision_frames: List, pixel_boost_total: int, model_size: Tuple[int, int], pixel_boost_size: Tuple[int, int]):
    """重组处理后的块为高分辨率图像"""
    crop_vision_frame = np.stack(temp_vision_frames).reshape(
        pixel_boost_total, pixel_boost_total, 
        model_size[1], model_size[0], 3
    )
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 3, 1, 4).reshape(
        pixel_boost_size[1], pixel_boost_size[0], 3
    )
    return crop_vision_frame

@lru_cache()
def get_static_model_initializer(model_path: str):
    """获取模型初始化器（用于inswapper）"""
    model = onnx.load(model_path)
    return onnx.numpy_helper.to_array(model.graph.initializer[-1])

def create_inference_session_providers(execution_device_id: str) -> List:
    """创建推理会话提供器 - 优化版"""
    if execution_device_id == "-1" or not torch.cuda.is_available():
        return ['CPUExecutionProvider']
    else:
        return [
            ('CUDAExecutionProvider', {
                'device_id': str(execution_device_id),
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_use_max_workspace': '0',
            }),
            'CPUExecutionProvider'
        ]

class InferencePool:
    """推理池管理器"""
    def __init__(self):
        self.sessions = {}
    
    def get(self, model_name: str):
        return self.sessions.get(model_name)
    
    def set(self, model_name: str, session):
        self.sessions[model_name] = session
# ===== GPUWorker 类 =====
class GPUWorker:
    """GPU工作器 - 处理人脸检测和换脸"""
    
    def __init__(self, gpu_id, source_faces, model_name='inswapper_128', max_age=70, 
                 sim_threshold=0.16, reset_interval=60, pixel_boost='128x128', 
                 auto_pixel_boost=False, frame_resolution=None, debug=False, swap_all_mode=False):
        self.gpu_id = gpu_id
        self.source_faces = source_faces
        self.model_name = model_name
        self.pixel_boost = pixel_boost
        self.auto_pixel_boost = auto_pixel_boost
        self.frame_resolution = frame_resolution
        self.models = None
        self.model_manager = ModelManager()
        self.inference_pool = InferencePool()
        
        # Swap-All 模式配置
        self.swap_all_mode = swap_all_mode
        if swap_all_mode:
            self.swap_all_source_face = source_faces['swap_face']  # 改名！
            self.skip_positions = set(source_faces['skip_positions'])
            if debug:
                print(f"[DEBUG] GPU {gpu_id} 启用 Swap-All 模式，跳过位置: {self.skip_positions}")
        
        # Track管理
        self.initial_mapping_done = False
        self.track_source_map = {}
        self.track_embeddings = {}
        self.frame_idx = 0
        self.max_age = max_age
        self.sim_threshold = sim_threshold
        self.reset_interval = reset_interval
        self.initial_frame_embeddings = {}
        self.lost_tracks = {}
        self.track_stability = {}
        
        # Skip机制
        self.skip_face_self_faces = {}  # {position: Face对象(自身)}
        
        # 表情容忍度
        self.embedding_history_size = 12
        self.expression_tolerance = {}
        self.debug = debug
        self.initialized = False
        
        # 初始化自动 pixel boost 选择器
        if self.auto_pixel_boost:
            model_type = 'hyperswap' if 'hyperswap' in model_name.lower() else 'inswapper'
            self.pixel_boost_selector = AutoPixelBoostSelector(model_type)
            if debug:
                print(f"[DEBUG] GPU {gpu_id} 启用自动 Pixel Boost (模型类型: {model_type})")
        else:
            self.pixel_boost_selector = None


    def get_model_options(self):
        """获取模型配置"""
        model_configs = {
            'inswapper_128': {
                'type': 'inswapper', 'template': 'arcface_128', 'size': (128, 128),
                'mean': [0.0, 0.0, 0.0], 'standard_deviation': [1.0, 1.0, 1.0],
                'pixel_boost_options': ['128x128', '256x256', '512x512']
            },
            'inswapper_128_fp16': {
                'type': 'inswapper', 'template': 'arcface_128', 'size': (128, 128),
                'mean': [0.0, 0.0, 0.0], 'standard_deviation': [1.0, 1.0, 1.0],
                'pixel_boost_options': ['128x128', '256x256', '512x512']
            },
            'hyperswap_1a_256': {
                'type': 'hyperswap', 'template': 'arcface_128', 'size': (256, 256),
                'mean': [0.5, 0.5, 0.5], 'standard_deviation': [0.5, 0.5, 0.5],
                'pixel_boost_options': ['256x256', '512x512', '768x768']
            },
            'hyperswap_1b_256': {
                'type': 'hyperswap', 'template': 'arcface_128', 'size': (256, 256),
                'mean': [0.5, 0.5, 0.5], 'standard_deviation': [0.5, 0.5, 0.5],
                'pixel_boost_options': ['256x256', '512x512', '768x768']
            },
            'hyperswap_1c_256': {
                'type': 'hyperswap', 'template': 'arcface_128', 'size': (256, 256),
                'mean': [0.5, 0.5, 0.5], 'standard_deviation': [0.5, 0.5, 0.5],
                'pixel_boost_options': ['256x256', '512x512', '768x768']
            }
        }
        return model_configs.get(self.model_name, model_configs['inswapper_128'])

    def initialize_models(self):
        """初始化模型"""
        if self.initialized and self.models:
            return True

        try:
            if self.debug:
                print(f"[GPU {self.gpu_id}] 开始初始化模型: {self.model_name}")
            
            # 创建推理会话提供器
            execution_device_id = str(self.gpu_id) if self.gpu_id >= 0 else "-1"
            providers = create_inference_session_providers(execution_device_id)
            
            if self.debug:
                print(f"[GPU {self.gpu_id}] 使用提供器: {providers}")
            
            # 显式为人脸检测器指定GPU
            ctx_id = self.gpu_id if self.gpu_id >= 0 else -1
            
            import insightface
            from core.config import get_face_model
            
            name = get_face_model()
            log_with_time("INFO", f"GPU {self.gpu_id} 初始化人脸检测器: {name} (ctx_id={ctx_id})")
            
            analyser = insightface.app.FaceAnalysis(name=name, providers=providers)
            analyser.prepare(ctx_id=ctx_id, det_size=(960, 960))
            log_with_time("INFO", f"GPU {self.gpu_id} 人脸检测器初始化完成")
            
            # 初始化换脸模型
            model_path = self.model_manager.get_model_path(self.model_name)
            if self.debug:
                print(f"[GPU {self.gpu_id}] 加载换脸模型: {model_path}")
            
            face_swapper_session = InferenceSession(model_path, providers=providers)
            self.inference_pool.set('face_swapper', face_swapper_session)
            self.models = {"analyser": analyser}

            # 验证source_faces
            if self.swap_all_mode:
                # Swap-All模式：验证单个源脸
                emb = getattr(self.swap_all_source_face, "embedding", None)  # 改名！
                model_src = getattr(self.swap_all_source_face, "source_model", getattr(self.swap_all_source_face, "detect_model", "unknown"))
                print(f"[INFO] Swap-All源人脸loaded: source_model={model_src}, emb_norm={np.linalg.norm(np.array(emb)) if emb is not None else 'N/A'}")
            else:
                # 正常模式：验证多个源脸
                for i, sf in enumerate(self.source_faces):
                    if sf is None:
                        print(f"[WARN] source_faces[{i}] is None")
                        continue
                    emb = getattr(sf, "embedding", None)
                    model_src = getattr(sf, "source_model", getattr(sf, "detect_model", "unknown"))
                    print(f"[INFO] source_faces[{i}] loaded: source_model={model_src}, emb_norm={np.linalg.norm(np.array(emb)) if emb is not None else 'N/A'}")

            self.initialized = True
            print(f"[INFO] GPU {self.gpu_id} 准备就绪，使用模型: {self.model_name}")
            return True

        except Exception as e:
            print(f"[ERROR] GPU {self.gpu_id} 初始化失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            self.initialized = False
            return False

        finally:
            try:
                cleanup_gpu_memory(self.gpu_id)
            except Exception as e:
                print(f"[WARN] cleanup_gpu_memory 调用失败: {e}")

    def prepare_source_embedding(self, source_face):
        """准备源人脸embedding"""
        model_options = self.get_model_options()
        model_type = model_options.get('type')
        
        if model_type == 'hyperswap':
            if hasattr(source_face, 'normed_embedding') and source_face.normed_embedding is not None:
                source_embedding = source_face.normed_embedding.reshape((1, -1))
            else:
                emb = source_face.embedding
                norm = np.linalg.norm(emb)
                normed_emb = emb / norm if norm > 0 else emb
                source_embedding = normed_emb.reshape((1, -1))
            return source_embedding
            
        elif model_type == 'inswapper':
            model_path = self.model_manager.get_model_path(self.model_name)
            model_initializer = get_static_model_initializer(model_path)
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = np.dot(source_embedding, model_initializer) / np.linalg.norm(source_embedding)
            return source_embedding
        
        return source_face.embedding.reshape((1, -1))

    def balance_source_embedding(self, source_embedding, target_embedding, face_swapper_weight=0.5):
        """平衡源和目标embedding"""
        model_options = self.get_model_options()
        model_type = model_options.get('type')
        
        face_swapper_weight = np.interp(face_swapper_weight, [0, 1], [0.35, -0.35]).astype(np.float32)
        
        if model_type in ['hyperswap', 'inswapper']:
            target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        source_embedding = source_embedding.reshape(1, -1)
        target_embedding = target_embedding.reshape(1, -1)
        source_embedding = source_embedding * (1 - face_swapper_weight) + target_embedding * face_swapper_weight
        return source_embedding

    def prepare_crop_frame(self, crop_vision_frame):
        """准备裁剪帧用于模型输入"""
        model_options = self.get_model_options()
        model_mean = model_options.get('mean')
        model_standard_deviation = model_options.get('standard_deviation')
        
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = np.expand_dims(crop_vision_frame, axis=0).astype(np.float32)
        return crop_vision_frame

    def normalize_crop_frame(self, crop_vision_frame):
        """标准化输出帧"""
        model_options = self.get_model_options()
        model_type = model_options.get('type')
        model_mean = model_options.get('mean')
        model_standard_deviation = model_options.get('standard_deviation')
        
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        
        if model_type in ['hyperswap']:
            crop_vision_frame = crop_vision_frame * model_standard_deviation + model_mean
        
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame

    def forward_swap_face(self, source_face, target_face, crop_vision_frame):
        """前向换脸推理"""
        face_swapper = self.inference_pool.get('face_swapper')
        
        face_swapper_inputs = {}
        
        for face_swapper_input in face_swapper.get_inputs():
            if face_swapper_input.name == 'source':
                source_embedding = self.prepare_source_embedding(source_face)
                source_embedding = self.balance_source_embedding(source_embedding, target_face.embedding)
                face_swapper_inputs[face_swapper_input.name] = source_embedding
            elif face_swapper_input.name == 'target':
                face_swapper_inputs[face_swapper_input.name] = crop_vision_frame
        
        with conditional_thread_semaphore():
            crop_vision_frame = face_swapper.run(None, face_swapper_inputs)[0][0]
        
        return crop_vision_frame

    def warp_face_by_face_landmark_5(self, temp_vision_frame, face_landmark_5, template, crop_size):
        """根据5点人脸关键点进行仿射变换"""
        template_points = np.array([
            [0.36167656, 0.40387734], [0.63696719, 0.40235469], 
            [0.50019687, 0.56044219], [0.38710391, 0.72160547],
            [0.61507734, 0.72034453]
        ]) * crop_size[0]
        
        affine_matrix = cv2.estimateAffinePartial2D(
            face_landmark_5, template_points, 
            method=cv2.RANSAC, ransacReprojThreshold=100
        )[0]
        
        crop_vision_frame = cv2.warpAffine(
            temp_vision_frame, affine_matrix, crop_size, 
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA
        )
        
        return crop_vision_frame, affine_matrix

    def paste_back(self, temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix, use_transparent_mask=False):
        """将处理后的人脸贴回原图"""
        if use_transparent_mask:
            if self.debug:
                print(f"[PASTE_BACK] 使用透明mask，保留原脸")
            return temp_vision_frame
        
        temp_height, temp_width = temp_vision_frame.shape[:2]
        crop_height, crop_width = crop_vision_frame.shape[:2]
        
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        
        crop_points = np.array([[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]])
        paste_points = cv2.transform(crop_points.reshape(1, -1, 2), inverse_matrix).reshape(-1, 2)
        
        paste_point_min = np.floor(paste_points.min(axis=0)).astype(int)
        paste_point_max = np.ceil(paste_points.max(axis=0)).astype(int)
        
        x1, y1 = np.clip(paste_point_min, 0, [temp_width, temp_height])
        x2, y2 = np.clip(paste_point_max, 0, [temp_width, temp_height])
        
        paste_width = x2 - x1
        paste_height = y2 - y1
        
        if paste_width <= 0 or paste_height <= 0:
            return temp_vision_frame
        
        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x1
        paste_matrix[1, 2] -= y1
        
        inverse_mask = cv2.warpAffine(crop_mask, paste_matrix, (paste_width, paste_height)).clip(0, 1)
        inverse_mask = np.expand_dims(inverse_mask, axis=-1)
        inverse_vision_frame = cv2.warpAffine(crop_vision_frame, paste_matrix, (paste_width, paste_height), borderMode=cv2.BORDER_REPLICATE)
        
        temp_vision_frame = temp_vision_frame.copy()
        paste_vision_frame = temp_vision_frame[y1:y2, x1:x2]
        paste_vision_frame = paste_vision_frame * (1 - inverse_mask) + inverse_vision_frame * inverse_mask
        temp_vision_frame[y1:y2, x1:x2] = paste_vision_frame.astype(temp_vision_frame.dtype)
        
        return temp_vision_frame

    def create_simple_mask(self, crop_vision_frame):
        """创建透明贴回用的自然面部 mask"""
        h, w = crop_vision_frame.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        
        max_r = 0.95 * min(h, w) / 2
        mask = np.clip(1 - (dist - max_r * 0.6) / (max_r * 0.4), 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), h * 0.03)
        
        return mask

    def swap_face(self, source_face, target_face, temp_vision_frame, is_skip=False):
        """换脸主函数 - 支持pixel boost超分和skip机制
        
        参数:
            source_face: 源人脸对象（要换成的脸）
            target_face: 目标人脸对象（视频中的脸）
            temp_vision_frame: 当前帧图像
            is_skip: 是否跳过换脸（使用透明mask）
        """
        model_options = self.get_model_options()
        model_template = model_options.get('template')
        model_size = model_options.get('size')

        # 自动选择 pixel boost
        if self.auto_pixel_boost and self.pixel_boost_selector is not None:
            recommended_boost = self.pixel_boost_selector.select_pixel_boost(
                target_face.bbox, self.frame_resolution
            )
            current_pixel_boost = recommended_boost
        
            if self.debug:
                face_size = self.pixel_boost_selector.calculate_face_size(target_face.bbox)
                print(f"[DEBUG] 人脸尺寸: {face_size:.1f}px, 选择 Pixel Boost: {recommended_boost}")
        else:
            current_pixel_boost = self.pixel_boost
        
        pixel_boost_size = unpack_resolution(current_pixel_boost)
        pixel_boost_total = pixel_boost_size[0] // model_size[0]
        
        # 验证pixel boost是否合法
        pixel_boost_options = model_options.get('pixel_boost_options', [])
        if current_pixel_boost not in pixel_boost_options:
            if self.debug:
                print(f"[DEBUG] Pixel boost {current_pixel_boost} 不支持，使用默认 {pixel_boost_options[0]}")
            pixel_boost_size = unpack_resolution(pixel_boost_options[0])
            pixel_boost_total = pixel_boost_size[0] // model_size[0]
        
        # 获取人脸关键点
        face_landmark_5 = target_face.kps
        if face_landmark_5 is None:
            if self.debug:
                print("[DEBUG] 目标人脸缺少关键点信息")
            return temp_vision_frame
        
        # 人脸对齐和裁剪
        crop_vision_frame, affine_matrix = self.warp_face_by_face_landmark_5(
            temp_vision_frame, face_landmark_5, model_template, pixel_boost_size
        )
        
        temp_vision_frames = []
        
        if pixel_boost_total > 1:
            # 使用pixel boost - 分块处理
            if self.debug:
                print(f"[DEBUG] 使用Pixel Boost: {pixel_boost_size} -> {model_size}, 分割为 {pixel_boost_total}x{pixel_boost_total} 块")
            
            pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
            
            for i, pixel_boost_vision_frame in enumerate(pixel_boost_vision_frames):
                pixel_boost_vision_frame_input = self.prepare_crop_frame(pixel_boost_vision_frame)
                pixel_boost_vision_frame_output = self.forward_swap_face(source_face, target_face, pixel_boost_vision_frame_input)
                pixel_boost_vision_frame_output = self.normalize_crop_frame(pixel_boost_vision_frame_output)
                temp_vision_frames.append(pixel_boost_vision_frame_output)
            
            crop_vision_frame_output = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)
            
        else:
            # 不使用pixel boost - 直接处理
            if self.debug:
                print(f"[DEBUG] 直接处理: {pixel_boost_size}")
            
            crop_vision_frame_input = self.prepare_crop_frame(crop_vision_frame)
            crop_vision_frame_output = self.forward_swap_face(source_face, target_face, crop_vision_frame_input)
            crop_vision_frame_output = self.normalize_crop_frame(crop_vision_frame_output)
        
        # 创建遮罩
        crop_mask = self.create_simple_mask(crop_vision_frame_output)
        
        if is_skip and self.debug:
            print(f"[SWAP_FACE] is_skip=True，将使用透明mask")
        
        # 贴回原图（skip人脸使用透明化）
        paste_vision_frame = self.paste_back(temp_vision_frame, crop_vision_frame_output, 
                                            crop_mask, affine_matrix, 
                                            use_transparent_mask=is_skip)
        
        return paste_vision_frame
    
    def _safe_swap_face(self, frame, target_face, source_face, debug_info="", track_id=None):
        """安全的换脸调用
        
        参数:
            frame: 当前帧图像
            target_face: 目标人脸对象（视频中的脸）
            source_face: 源人脸对象（要换成的脸，skip时传入目标脸自己）
            debug_info: 调试信息
            track_id: track ID
        """
        try:
            # 检查是否是SKIP track
            is_skip = False
            if track_id is not None:
                if self.swap_all_mode:
                    # Swap-All模式：检查track_source_map
                    is_skip = self.track_source_map.get(track_id) == "SKIP"
                else:
                    # 正常模式：检查skip_face_self_faces
                    is_skip = track_id in self.skip_face_self_faces
            
            if is_skip and self.debug:
                print(f"[DEBUG] Track {track_id} 是SKIP，执行换脸但透明化贴回")
            
            # 注意：SKIP也要执行换脸（自己换自己），但最后透明化贴回
            # 这样可以保持track的连续性，避免被其他人脸误匹配
            return self.swap_face(source_face, target_face, frame, is_skip=is_skip)
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] {debug_info} 换脸失败: {e}")
                import traceback
                traceback.print_exc()
            return None
        
# ===== GPUWorker Tracking 逻辑 (续) =====
    
    def _get_embedding(self, face):
        """获取人脸embedding"""
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        
        # 尝试触发懒加载或计算
        if emb is None:
            try:
                _ = getattr(face, "embedding", None)
                emb = getattr(face, "normed_embedding", None) or getattr(face, "embedding", None)
            except Exception:
                emb = None
        
        if emb is None:
            return None
        
        arr = np.array(emb, dtype=np.float32)
        # 过滤 NaN / 非法向量
        if np.isnan(arr).any() or np.linalg.norm(arr) == 0:
            return None
        return arr

    def _get_average_embedding(self, track_id):
        """获取track的平均embedding"""
        if track_id not in self.track_embeddings:
            return None
        
        history = self.track_embeddings[track_id].get("history", [])
        current_emb = self.track_embeddings[track_id]["emb"]
        
        if not history:
            return current_emb
        
        all_embs = history + [current_emb]
        weights = np.linspace(0.5, 1.0, len(all_embs))
        weighted_sum = np.zeros_like(current_emb)
        weight_sum = 0
        
        for emb, weight in zip(all_embs, weights):
            weighted_sum += emb * weight
            weight_sum += weight
            
        avg_emb = weighted_sum / weight_sum
        norm = np.linalg.norm(avg_emb)
        return avg_emb / norm if norm > 0 else avg_emb

    def _update_embedding_history(self, track_id, new_emb):
        """更新embedding历史"""
        if track_id not in self.track_embeddings:
            return
        
        history = self.track_embeddings[track_id].get("history", [])
        history.append(new_emb.copy())
        
        if len(history) > self.embedding_history_size:
            history = history[-self.embedding_history_size:]
        
        self.track_embeddings[track_id]["history"] = history

    def _is_track_stable(self, track_id, min_frames=3):
        """判断track是否稳定"""
        if track_id not in self.track_stability:
            self.track_stability[track_id] = 1
            return min_frames <= 1
        
        return self.track_stability[track_id] >= min_frames

    def _increment_stability(self, track_id):
        """增加稳定性计数"""
        if track_id not in self.track_stability:
            self.track_stability[track_id] = 1
        else:
            self.track_stability[track_id] += 1

    def _adaptive_similarity_threshold(self, base_emb, candidate_emb, track_id=None):
        """自适应相似度阈值"""
        if track_id is not None:
            src_idx = self.track_source_map.get(track_id, None)
            if src_idx is None:
                return False, -1.0
        
        base_sim = cosine_sim(base_emb, candidate_emb)
        
        if base_sim < 0:
            return False, base_sim
        
        threshold = max(self.sim_threshold, 0.18)
        
        # 自适应调整阈值
        if track_id is not None and track_id in self.track_embeddings:
            history = self.track_embeddings[track_id].get("history", [])
            original_emb = self.track_embeddings[track_id].get("original_emb")
            
            if len(history) >= 3 and original_emb is not None:
                original_sim = cosine_sim(candidate_emb, original_emb)
                
                # 检测表情变化
                if len(history) >= 5:
                    frame_to_frame_sims = []
                    for i in range(len(history)-1):
                        sim = cosine_sim(history[i], history[i+1])
                        if sim > 0:
                            frame_to_frame_sims.append(sim)
                    
                    if frame_to_frame_sims:
                        avg_frame_sim = np.mean(frame_to_frame_sims)
                        sim_variance = np.var(frame_to_frame_sims)
                        
                        if sim_variance > 0.02:
                            threshold *= 0.85
                            if self.debug:
                                print(f"[DEBUG] Track {track_id} 检测到剧烈变化（表情），放宽阈值至 {threshold:.3f}")
                        elif sim_variance > 0.01:
                            threshold *= 0.90
                        elif avg_frame_sim < threshold + 0.03:
                            threshold *= 0.93
                
                # 与原始embedding比较
                if original_sim > 0:
                    if original_sim > threshold + 0.1:
                        threshold *= 0.8
                        if self.debug:
                            print(f"[DEBUG] Track {track_id} 与原始高度相似，放宽阈值至 {threshold:.3f}")
                    elif original_sim > threshold + 0.05:
                        threshold *= 0.85
                    
                    # 回归原始趋势
                    if len(history) >= 3:
                        recent_original_sims = [cosine_sim(h, original_emb) for h in history[-3:]]
                        recent_original_sims = [s for s in recent_original_sims if s > 0]
                        if recent_original_sims:
                            trend = original_sim - np.mean(recent_original_sims)
                            if trend > 0.03:
                                threshold *= 0.85
            
            # 表情容忍度累积
            if track_id not in self.expression_tolerance:
                self.expression_tolerance[track_id] = 0
            
            near_threshold_range = 0.06
            if base_sim > threshold - near_threshold_range and base_sim < threshold + 0.02:
                self.expression_tolerance[track_id] += 1
                if self.expression_tolerance[track_id] >= 2:
                    extra_reduction = min(0.15, self.expression_tolerance[track_id] * 0.03)
                    threshold *= (1 - extra_reduction)
                    if self.debug:
                        print(f"[DEBUG] Track {track_id} 累积容忍度 {self.expression_tolerance[track_id]}, 放宽阈值至 {threshold:.3f}")
            elif base_sim > threshold + 0.02:
                self.expression_tolerance[track_id] = 0
        
        threshold = max(threshold, 0.15)
        
        # 多级阈值判断
        very_high_threshold = threshold + 0.06
        high_threshold = threshold + 0.03
        medium_threshold = threshold
        low_threshold = threshold - 0.03
        very_low_threshold = threshold - 0.06
        
        if base_sim >= very_high_threshold:
            return True, base_sim
        elif base_sim >= high_threshold:
            return True, base_sim
        elif base_sim >= medium_threshold:
            if track_id is not None and self._is_track_stable(track_id, min_frames=1):
                return True, base_sim
            return False, base_sim
        elif base_sim >= low_threshold:
            if track_id is not None and self._is_track_stable(track_id, min_frames=2):
                return True, base_sim
            return False, base_sim
        elif base_sim >= very_low_threshold:
            if track_id is not None and self._is_track_stable(track_id, min_frames=3):
                return True, base_sim
            return False, base_sim
        else:
            return False, base_sim

    def _prune_tracks(self):
        """清理过期的tracks"""
        if self.max_age == -1:
            return
            
        current_tracks = {}    
        for tid, t in self.track_embeddings.items():
            # ghost skip track 永不轻易删除
            if self.track_source_map.get(tid) == "SKIP":
                current_tracks[tid] = t
                continue
            
            if (self.frame_idx - t["last_seen"]) <= self.max_age:
                current_tracks[tid] = t
            else:
                if tid not in self.lost_tracks:
                    self.lost_tracks[tid] = {
                        "emb": t.get("original_emb", t["emb"]),
                        "source_idx": self.track_source_map.get(tid),
                        "lost_frame": self.frame_idx,
                        "history": t.get("history", [])
                    }
                    if self.debug:
                        print(f"[DEBUG] Track {tid} 移入丢失列表")
                self.track_stability.pop(tid, None)
                
                # 清理skip缓存
                if tid not in current_tracks:
                    self.skip_face_self_faces.pop(tid, None)
        
        self.track_embeddings = current_tracks
        
        # 清理长期丢失的tracks
        if self.max_age > 0:
            max_lost_age = self.max_age * 8
            self.lost_tracks = {tid: t for tid, t in self.lost_tracks.items() 
                               if (self.frame_idx - t["lost_frame"]) <= max_lost_age}

    def _should_reset_tracking(self):
        """判断是否应该重置tracking"""
        return (self.frame_idx > 0 and 
                self.frame_idx % self.reset_interval == 0 and 
                self.initial_frame_embeddings)

    def _reset_tracking_with_reference(self, faces):
        """使用参考帧重置tracking"""
        if not self.initial_frame_embeddings:
            return False
            
        if self.debug:
            print(f"[DEBUG] 帧 {self.frame_idx}: 重新检索匹配")
        
        faces_sorted = sorted(faces, key=lambda x: x.bbox[0])
        reset_success = False
        
        sims = []
        for det_idx, face in enumerate(faces_sorted):
            emb = self._get_embedding(face)
            if emb is None:
                continue            

            for ref_idx, ref_emb in self.initial_frame_embeddings.items():
                # Swap-All模式：验证ref_idx是否在跳过位置
                if self.swap_all_mode:
                    # 跳过位置匹配自己，其他位置匹配源脸
                    pass  # 继续匹配逻辑
                else:
                    # 正常模式：验证ref_idx范围
                    if ref_idx >= len(self.source_faces):
                        continue
                
                sim = cosine_sim(emb, ref_emb)
                sims.append((sim, det_idx, ref_idx, emb))
        
        sims.sort(key=lambda x: x[0], reverse=True)
        used_faces, used_sources = set(), set()
        new_mappings = {}
        
        reset_threshold = self.sim_threshold * 0.6
        
        for sim, det_idx, ref_idx, emb in sims:
            if sim < reset_threshold:
                continue
            if det_idx in used_faces or ref_idx in used_sources:
                continue
                
            track_id = ref_idx
            new_mappings[track_id] = {
                "emb": emb, 
                "last_seen": self.frame_idx,
                "original_emb": self.initial_frame_embeddings[ref_idx],
                "history": []
            }
            
            # Swap-All模式：根据位置决定映射类型
            if self.swap_all_mode:
                if ref_idx in self.skip_positions:
                    self.track_source_map[track_id] = "SKIP"
                    self.skip_face_self_faces[ref_idx] = faces_sorted[det_idx]
                else:
                    self.track_source_map[track_id] = "SWAP"
            else:
                # 正常模式
                self.track_source_map[track_id] = ref_idx
                if ref_idx in self.skip_face_self_faces:
                    self.skip_face_self_faces[ref_idx] = faces_sorted[det_idx]
            
            self.track_stability[track_id] = 2
            used_faces.add(det_idx)
            used_sources.add(ref_idx)
            reset_success = True
            
            if self.debug:
                if self.swap_all_mode:
                    skip_info = " (SKIP)" if ref_idx in self.skip_positions else ""
                else:
                    skip_info = " (SKIP)" if ref_idx in self.skip_face_self_faces else ""
                print(f"[DEBUG] 重置: Track {track_id} <- 源 {ref_idx} sim={sim:.3f}{skip_info}")
        
        # 保留未匹配但未过期的track
        for tid, t in self.track_embeddings.items():
            if tid not in new_mappings and (self.frame_idx - t["last_seen"]) <= 5:
                new_mappings[tid] = t
        
        self.track_embeddings = new_mappings
        return reset_success

    def _try_recover_lost_tracks(self, faces):
        """尝试恢复丢失的tracks"""
        if not self.lost_tracks:
            return []
            
        faces_sorted = sorted(faces, key=lambda x: x.bbox[0])
        recovered = []
        
        for det_idx, face in enumerate(faces_sorted):
            emb = self._get_embedding(face)
            if emb is None:
                continue
                
            best_sim, best_tid = -1, None          
            for tid, lost_info in self.lost_tracks.items():
                # swap-all模式和正常模式都一样处理
                if self.swap_all_mode:
                    src_type = lost_info.get("source_idx")
                    if src_type == "SKIP":
                        # SKIP track：与自己的历史embedding比较
                        sims_to_check = [lost_info["emb"]]
                        sims_to_check.extend(lost_info.get("history", [])[-3:])
                        best_self_sim = max([cosine_sim(emb, e) for e in sims_to_check])
                        if best_self_sim > self.sim_threshold * 0.75:
                            best_tid = tid
                            best_sim = best_self_sim
                        continue
                else:
                    src_idx = lost_info.get("source_idx")
                    if src_idx is None or (isinstance(src_idx, int) and src_idx >= len(self.source_faces)):
                        # 这是个SKIP track（source_idx可能是None，或者source_faces[src_idx]是None）
                        sims_to_check = [lost_info["emb"]]
                        sims_to_check.extend(lost_info.get("history", [])[-3:])
                        best_self_sim = max([cosine_sim(emb, e) for e in sims_to_check])
                        if best_self_sim > self.sim_threshold * 0.75:
                            best_tid = tid
                            best_sim = best_self_sim
                        continue
                
                # 非SKIP的原逻辑
                sims_to_check = [lost_info["emb"]]
                if "history" in lost_info and lost_info["history"]:
                    sims_to_check.extend(lost_info["history"][-2:])
                            
                max_sim = max([cosine_sim(emb, check_emb) for check_emb in sims_to_check])
                            
                if max_sim > best_sim and max_sim > self.sim_threshold * 0.7:
                    best_sim, best_tid = max_sim, tid
            
            if best_tid is not None:
                lost_info = self.lost_tracks.pop(best_tid)
                self.track_embeddings[best_tid] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": lost_info["emb"],
                    "history": lost_info.get("history", [])
                }
                self.track_source_map[best_tid] = lost_info["source_idx"]
                
                # 恢复skip track时也同步skip记忆
                if self.swap_all_mode:
                    if lost_info["source_idx"] == "SKIP":
                        self.skip_face_self_faces[best_tid] = face
                else:
                    if lost_info["source_idx"] in self.skip_face_self_faces or (
                        isinstance(lost_info["source_idx"], int) and 
                        lost_info["source_idx"] < len(self.source_faces) and 
                        self.source_faces[lost_info["source_idx"]] is None
                    ):
                        self.skip_face_self_faces[best_tid] = face

                old_stability = lost_info.get("stability", 1)
                self.track_stability[best_tid] = min(old_stability, 5)
                recovered.append((best_tid, det_idx, best_sim))
                
                if self.debug:
                    skip_info = " (SKIP)" if (
                        self.swap_all_mode and lost_info["source_idx"] == "SKIP"
                    ) or (
                        not self.swap_all_mode and best_tid in self.skip_face_self_faces
                    ) else ""
                    print(f"[DEBUG] 恢复 Track {best_tid}{skip_info} sim={best_sim:.3f}")
        
        return recovered

    def process_frame(self, frame_path):
        """处理单帧 - 主入口（支持swap-all模式）"""
        if not self.models:
            return False, "模型未初始化"

        frame = cv2.imread(frame_path)
        if frame is None:
            return False, f"无法读取 {frame_path}"

        faces = self.models["analyser"].get(frame)
        if not faces:
            cv2.imwrite(frame_path, frame)
            self.frame_idx += 1
            self._prune_tracks()
            return True, "无人脸"

        faces_sorted = sorted(faces, key=lambda x: x.bbox[0])

        if self.swap_all_mode:
            # Swap-All 模式处理
            return self._process_frame_swap_all(frame, faces_sorted, frame_path)
        else:
            # 正常模式处理
            return self._process_frame_normal(frame, faces_sorted, frame_path)

        # 第一帧初始化
        if not self.initial_mapping_done:
            if self.debug:
                print(f"[DEBUG] GPU {self.gpu_id} 处理第一帧，检测到 {len(faces_sorted)} 个人脸")
            
            processed_frame = frame.copy()
            
            for i, face in enumerate(faces_sorted):
                if i >= len(self.source_faces):
                    break
                
                src_face = self.source_faces[i]
                emb = self._get_embedding(face)
                if emb is None:
                    continue

                # 所有人脸都建立 track
                self.initial_frame_embeddings[i] = emb.copy()
                self.track_embeddings[i] = {
                    "emb": emb, 
                    "last_seen": self.frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                self.track_stability[i] = 1
                self.track_source_map[i] = i
            
                if src_face is None:
                    # skip人脸：保存自身Face对象
                    self.skip_face_self_faces[i] = face
                    if self.debug:
                        print(f"[DEBUG] GPU {self.gpu_id} Track {i} 设为SKIP (使用自身)")
                else:
                    # 正常换脸
                    newf = self._safe_swap_face(processed_frame, face, src_face, 
                                               f"GPU {self.gpu_id} 第一帧 Track {i}", 
                                               track_id=i)
                    if newf is not None:
                        processed_frame = newf
                        if self.debug:
                            print(f"[DEBUG] GPU {self.gpu_id} 第一帧: Track {i} <- 源 {i} 成功")
            
            frame = processed_frame
            self.initial_mapping_done = True

        else:
            # 后续帧处理
            if self._should_reset_tracking():
                if self._reset_tracking_with_reference(faces):
                    # 重置后立即处理
                    for tid, t in self.track_embeddings.items():
                        if tid not in self.track_source_map:
                            continue
                        src_idx = self.track_source_map[tid]
                        src_face = self.source_faces[src_idx]
                        if src_face is None:
                            continue
                            
                        for det_idx, face in enumerate(faces_sorted):
                            face_emb = self._get_embedding(face)
                            if face_emb is not None and cosine_sim(face_emb, t["emb"]) > self.sim_threshold:
                                newf = self._safe_swap_face(frame, face, src_face, f"重置后 Track {tid}", track_id=tid)
                                if newf is not None:
                                    frame = newf
                                    if self.debug:
                                        print(f"[DEBUG] 重置后替换: Track {tid} <- 源 {src_idx}")
                                break
                    
                    cv2.imwrite(frame_path, frame)
                    self.frame_idx += 1
                    self._prune_tracks()
                    return True, "重置完成"
            
            self._try_recover_lost_tracks(faces)
            
            # 匹配所有track
            sims = []
            for det_idx, face in enumerate(faces_sorted):
                emb = self._get_embedding(face)
                if emb is None:
                    continue
                
                for tid, t in self.track_embeddings.items():
                    if tid not in self.track_source_map:
                        continue
                    
                    avg_emb = self._get_average_embedding(tid)
                    is_match, sim = self._adaptive_similarity_threshold(avg_emb, emb, tid)
                    
                    if is_match:
                        sims.append((sim, det_idx, tid, emb))

            sims.sort(key=lambda x: x[0], reverse=True)
            used_tracks, used_faces = set(), set()
            new_embeddings = {}

            for sim, det_idx, tid, emb in sims:
                if tid in used_tracks or det_idx in used_faces:
                    continue

                src_idx = self.track_source_map.get(tid, None)
                if src_idx is None:
                    continue
                
                # 获取源人脸（SKIP的用当前检测到的Face）
                is_skip_track = tid in self.skip_face_self_faces
                
                if is_skip_track:
                    if det_idx < len(faces_sorted):
                        current_face = faces_sorted[det_idx]
                        src_face = current_face
                        self.skip_face_self_faces[tid] = current_face
                    else:
                        continue
                else:
                    src_face = self.source_faces[src_idx]
                
                if src_face is None:
                    continue
                
                # 判断是否应该换脸
                should_swap = (
                    sim > self.sim_threshold + 0.06 or
                    sim > self.sim_threshold + 0.03 or
                    (sim > self.sim_threshold and self._is_track_stable(tid, 1)) or
                    (sim > self.sim_threshold - 0.03 and self._is_track_stable(tid, 2))
                )
                
                if should_swap:
                    newf = self._safe_swap_face(frame, faces_sorted[det_idx], 
                                               src_face, f"GPU{self.gpu_id}", track_id=tid)
                    if newf is not None:
                        frame = newf
                        self._increment_stability(tid)
                        if self.debug:
                            skip_info = " (SKIP)" if is_skip_track else ""
                            print(f"[DEBUG] 帧{self.frame_idx} Track{tid}{skip_info} 替换成功 sim={sim:.3f}")
                elif sim > self.sim_threshold - 0.03:
                    self._increment_stability(tid)
                
                # 更新track（SKIP和正常都需要）
                old_track = self.track_embeddings.get(tid, {})
                new_embeddings[tid] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": old_track.get("original_emb", emb),
                    "history": old_track.get("history", [])
                }
                self._update_embedding_history(tid, emb)
                
                used_tracks.add(tid)
                used_faces.add(det_idx)
            
            # 保留未匹配但未过期的track
            for tid, t in self.track_embeddings.items():
                if tid not in new_embeddings:
                    if self.max_age == -1 or (self.frame_idx - t["last_seen"]) <= self.max_age:
                        new_embeddings[tid] = t
                        if tid in self.track_stability and self.track_stability[tid] > 1:
                            self.track_stability[tid] = max(1, self.track_stability[tid] - 1)
            
            self.track_embeddings = new_embeddings

        cv2.imwrite(frame_path, frame)
        self.frame_idx += 1
        self._prune_tracks()
        return True, "完成"
    def _process_frame_swap_all(self, frame, faces_sorted, frame_path):
        """Swap-All模式的帧处理"""
        processed_frame = frame.copy()
        
        # 第一帧初始化
        if not self.initial_mapping_done:
            if self.debug:
                print(f"[DEBUG] GPU {self.gpu_id} Swap-All第一帧，检测到 {len(faces_sorted)} 个人脸")
            
            for i, face in enumerate(faces_sorted):
                emb = self._get_embedding(face)
                if emb is None:
                    continue
                
                # 所有人脸都建立track
                self.initial_frame_embeddings[i] = emb.copy()
                self.track_embeddings[i] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                self.track_stability[i] = 1
                
                if i in self.skip_positions:
                    # 跳过位置：标记为SKIP，使用自己换自己+透明化
                    self.track_source_map[i] = "SKIP"
                    self.skip_face_self_faces[i] = face
                    
                    # 执行换脸（自己换自己），但会透明化贴回
                    newf = self._safe_swap_face(processed_frame, face, face,  # 注意：源脸也是face自己
                                               f"GPU {self.gpu_id} 第一帧 Track {i}",
                                               track_id=i)
                    if newf is not None:
                        processed_frame = newf
                        if self.debug:
                            print(f"[DEBUG] GPU {self.gpu_id} Track {i} SKIP（自己换自己+透明化）")
                else:
                    # 其他位置：换成源脸
                    self.track_source_map[i] = "SWAP"
                    newf = self._safe_swap_face(processed_frame, face, self.swap_face,
                                               f"GPU {self.gpu_id} 第一帧 Track {i}",
                                               track_id=i)
                    if newf is not None:
                        processed_frame = newf
                        if self.debug:
                            print(f"[DEBUG] GPU {self.gpu_id} 第一帧: Track {i} 换脸成功")
            
            frame = processed_frame
            self.initial_mapping_done = True
        
        else:
            # 后续帧处理
            if self._should_reset_tracking():
                if self._reset_tracking_with_reference(faces_sorted):
                    for tid, t in self.track_embeddings.items():
                        src_type = self.track_source_map.get(tid)
                        
                        for det_idx, face in enumerate(faces_sorted):
                            face_emb = self._get_embedding(face)
                            if face_emb is not None and cosine_sim(face_emb, t["emb"]) > self.sim_threshold:
                                if src_type == "SKIP":
                                    # SKIP: 自己换自己+透明化
                                    newf = self._safe_swap_face(frame, face, face,
                                                               f"重置后 Track {tid}", track_id=tid)
                                else:
                                    # SWAP: 换成源脸
                                    newf = self._safe_swap_face(frame, face, self.swap_face,
                                                               f"重置后 Track {tid}", track_id=tid)
                                
                                if newf is not None:
                                    frame = newf
                                    if self.debug:
                                        print(f"[DEBUG] 重置后替换: Track {tid}")
                                break
                    
                    cv2.imwrite(frame_path, frame)
                    self.frame_idx += 1
                    self._prune_tracks()
                    return True, "重置完成"
            
            self._try_recover_lost_tracks(faces_sorted)
            
            # 匹配所有track
            sims = []
            for det_idx, face in enumerate(faces_sorted):
                emb = self._get_embedding(face)
                if emb is None:
                    continue
                
                for tid, t in self.track_embeddings.items():
                    avg_emb = self._get_average_embedding(tid)
                    is_match, sim = self._adaptive_similarity_threshold(avg_emb, emb, tid)
                    
                    if is_match:
                        sims.append((sim, det_idx, tid, emb))
            
            sims.sort(key=lambda x: x[0], reverse=True)
            used_tracks, used_faces = set(), set()
            new_embeddings = {}
            
            # 处理匹配的人脸
            for sim, det_idx, tid, emb in sims:
                if tid in used_tracks or det_idx in used_faces:
                    continue
                
                src_type = self.track_source_map.get(tid)
                if src_type is None:
                    continue
                
                # 判断是否应该换脸
                should_swap = (
                    sim > self.sim_threshold + 0.06 or
                    sim > self.sim_threshold + 0.03 or
                    (sim > self.sim_threshold and self._is_track_stable(tid, 1)) or
                    (sim > self.sim_threshold - 0.03 and self._is_track_stable(tid, 2))
                )
                
                if should_swap:
                    current_face = faces_sorted[det_idx]
                    
                    if src_type == "SKIP":
                        # SKIP track：自己换自己+透明化
                        self.skip_face_self_faces[tid] = current_face
                        newf = self._safe_swap_face(frame, current_face, current_face,
                                                   f"GPU{self.gpu_id}", track_id=tid)
                    else:
                        # SWAP track：换成源脸
                        newf = self._safe_swap_face(frame, current_face, self.swap_face,
                                                   f"GPU{self.gpu_id}", track_id=tid)
                    
                    if newf is not None:
                        frame = newf
                        self._increment_stability(tid)
                        if self.debug:
                            skip_info = " (SKIP)" if src_type == "SKIP" else ""
                            print(f"[DEBUG] 帧{self.frame_idx} Track{tid}{skip_info} 替换成功 sim={sim:.3f}")
                elif sim > self.sim_threshold - 0.03:
                    self._increment_stability(tid)
                    if src_type == "SKIP" and det_idx < len(faces_sorted):
                        self.skip_face_self_faces[tid] = faces_sorted[det_idx]
                
                # 更新track
                old_track = self.track_embeddings.get(tid, {})
                new_embeddings[tid] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": old_track.get("original_emb", emb),
                    "history": old_track.get("history", [])
                }
                self._update_embedding_history(tid, emb)
                
                used_tracks.add(tid)
                used_faces.add(det_idx)
            
            # 处理未匹配的新人脸（track之外出现的人脸）
            for det_idx, face in enumerate(faces_sorted):
                if det_idx in used_faces:
                    continue
                
                emb = self._get_embedding(face)
                if emb is None:
                    continue
                
                # 创建新track，默认换成源脸
                new_tid = max(self.track_embeddings.keys()) + 1 if self.track_embeddings else 0
                
                self.track_embeddings[new_tid] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                self.track_source_map[new_tid] = "SWAP"
                self.track_stability[new_tid] = 1
                
                # 立即换脸
                newf = self._safe_swap_face(frame, face, self.swap_face,
                                           f"GPU{self.gpu_id} 新人脸", track_id=new_tid)
                if newf is not None:
                    frame = newf
                    if self.debug:
                        print(f"[DEBUG] 帧{self.frame_idx} 新人脸 Track{new_tid} 换脸成功")
            
            # 保留未匹配但未过期的track
            for tid, t in self.track_embeddings.items():
                if tid not in new_embeddings:
                    if self.max_age == -1 or (self.frame_idx - t["last_seen"]) <= self.max_age:
                        new_embeddings[tid] = t
                        if tid in self.track_stability and self.track_stability[tid] > 1:
                            self.track_stability[tid] = max(1, self.track_stability[tid] - 1)
            
            self.track_embeddings = new_embeddings
        
        cv2.imwrite(frame_path, frame)
        self.frame_idx += 1
        self._prune_tracks()
        return True, "完成"


    def _process_frame_normal(self, frame, faces_sorted, frame_path):
        """正常模式的帧处理（原有逻辑）"""
        processed_frame = frame.copy()
        
        # 第一帧初始化
        if not self.initial_mapping_done:
            if self.debug:
                print(f"[DEBUG] GPU {self.gpu_id} 处理第一帧，检测到 {len(faces_sorted)} 个人脸")
            
            for i, face in enumerate(faces_sorted):
                if i >= len(self.source_faces):
                    break
                
                src_face = self.source_faces[i]
                emb = self._get_embedding(face)
                if emb is None:
                    continue

                # 所有人脸都建立 track
                self.initial_frame_embeddings[i] = emb.copy()
                self.track_embeddings[i] = {
                    "emb": emb, 
                    "last_seen": self.frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                self.track_stability[i] = 1
                self.track_source_map[i] = i
            
                if src_face is None:
                    # skip人脸：保存自身Face对象
                    self.skip_face_self_faces[i] = face
                    if self.debug:
                        print(f"[DEBUG] GPU {self.gpu_id} Track {i} 设为SKIP (使用自身)")
                else:
                    # 正常换脸
                    newf = self._safe_swap_face(processed_frame, face, src_face, 
                                               f"GPU {self.gpu_id} 第一帧 Track {i}", 
                                               track_id=i)
                    if newf is not None:
                        processed_frame = newf
                        if self.debug:
                            print(f"[DEBUG] GPU {self.gpu_id} 第一帧: Track {i} <- 源 {i} 成功")
            
            frame = processed_frame
            self.initial_mapping_done = True

        else:
            # 后续帧处理
            if self._should_reset_tracking():
                if self._reset_tracking_with_reference(faces_sorted):
                    # 重置后立即处理
                    for tid, t in self.track_embeddings.items():
                        if tid not in self.track_source_map:
                            continue
                        src_idx = self.track_source_map[tid]
                        src_face = self.source_faces[src_idx]
                        if src_face is None:
                            continue
                            
                        for det_idx, face in enumerate(faces_sorted):
                            face_emb = self._get_embedding(face)
                            if face_emb is not None and cosine_sim(face_emb, t["emb"]) > self.sim_threshold:
                                newf = self._safe_swap_face(frame, face, src_face, f"重置后 Track {tid}", track_id=tid)
                                if newf is not None:
                                    frame = newf
                                    if self.debug:
                                        print(f"[DEBUG] 重置后替换: Track {tid} <- 源 {src_idx}")
                                break
                    
                    cv2.imwrite(frame_path, frame)
                    self.frame_idx += 1
                    self._prune_tracks()
                    return True, "重置完成"
            
            self._try_recover_lost_tracks(faces_sorted)
            
            # 匹配所有track
            sims = []
            for det_idx, face in enumerate(faces_sorted):
                emb = self._get_embedding(face)
                if emb is None:
                    continue
                
                for tid, t in self.track_embeddings.items():
                    if tid not in self.track_source_map:
                        continue
                    
                    avg_emb = self._get_average_embedding(tid)
                    is_match, sim = self._adaptive_similarity_threshold(avg_emb, emb, tid)
                    
                    if is_match:
                        sims.append((sim, det_idx, tid, emb))

            sims.sort(key=lambda x: x[0], reverse=True)
            used_tracks, used_faces = set(), set()
            new_embeddings = {}

            for sim, det_idx, tid, emb in sims:
                if tid in used_tracks or det_idx in used_faces:
                    continue

                src_idx = self.track_source_map.get(tid, None)
                if src_idx is None:
                    continue
                
                # 获取源人脸（SKIP的用当前检测到的Face）
                is_skip_track = tid in self.skip_face_self_faces
                
                if is_skip_track:
                    if det_idx < len(faces_sorted):
                        current_face = faces_sorted[det_idx]
                        src_face = current_face
                        self.skip_face_self_faces[tid] = current_face
                    else:
                        continue
                else:
                    src_face = self.source_faces[src_idx]
                
                if src_face is None:
                    continue
                
                # 判断是否应该换脸
                should_swap = (
                    sim > self.sim_threshold + 0.06 or
                    sim > self.sim_threshold + 0.03 or
                    (sim > self.sim_threshold and self._is_track_stable(tid, 1)) or
                    (sim > self.sim_threshold - 0.03 and self._is_track_stable(tid, 2))
                )
                
                if should_swap:
                    newf = self._safe_swap_face(frame, faces_sorted[det_idx], 
                                               src_face, f"GPU{self.gpu_id}", track_id=tid)
                    if newf is not None:
                        frame = newf
                        self._increment_stability(tid)
                        if self.debug:
                            skip_info = " (SKIP)" if is_skip_track else ""
                            print(f"[DEBUG] 帧{self.frame_idx} Track{tid}{skip_info} 替换成功 sim={sim:.3f}")
                elif sim > self.sim_threshold - 0.03:
                    self._increment_stability(tid)
                
                # 更新track（SKIP和正常都需要）
                old_track = self.track_embeddings.get(tid, {})
                new_embeddings[tid] = {
                    "emb": emb,
                    "last_seen": self.frame_idx,
                    "original_emb": old_track.get("original_emb", emb),
                    "history": old_track.get("history", [])
                }
                self._update_embedding_history(tid, emb)
                
                used_tracks.add(tid)
                used_faces.add(det_idx)
            
            # 保留未匹配但未过期的track
            for tid, t in self.track_embeddings.items():
                if tid not in new_embeddings:
                    if self.max_age == -1 or (self.frame_idx - t["last_seen"]) <= self.max_age:
                        new_embeddings[tid] = t
                        if tid in self.track_stability and self.track_stability[tid] > 1:
                            self.track_stability[tid] = max(1, self.track_stability[tid] - 1)
            
            self.track_embeddings = new_embeddings

        cv2.imwrite(frame_path, frame)
        self.frame_idx += 1
        self._prune_tracks()
        return True, "完成"
    
# ===== 主处理函数 =====

def _process_frame_with_worker(worker, frame, faces, frame_idx, debug):
    """处理单帧 - 使用透明mask方案"""
    faces_sorted = sorted(faces, key=lambda x: x.bbox[0])
    processed_frame = frame.copy()
    
    if not worker.initial_mapping_done:
        # 第一帧处理
        if worker.swap_all_mode:
            # Swap-All模式
            for i, face in enumerate(faces_sorted):
                emb = worker._get_embedding(face)
                
                if emb is None:
                    continue
                
                # 所有人脸都建立track
                worker.initial_frame_embeddings[i] = emb.copy()
                worker.track_embeddings[i] = {
                    "emb": emb,
                    "last_seen": frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                worker.track_stability[i] = 1
                
                if i in worker.skip_positions:
                    # 跳过位置：使用自己换自己+透明化
                    worker.track_source_map[i] = "SKIP"
                    worker.skip_face_self_faces[i] = face
                    
                    # 自己换自己，但会透明化贴回
                    newf = worker._safe_swap_face(processed_frame, face, face,
                                                 f"GPU{worker.gpu_id}", track_id=i)
                    if newf is not None:
                        processed_frame = newf
                    
                    if debug:
                        from core.checkpoint_manager import log_with_time
                        log_with_time("DEBUG", f"Track {i} 标记为SKIP（自己换自己+透明化）")
                else:
                    # 换脸位置
                    worker.track_source_map[i] = "SWAP"
                    newf = worker._safe_swap_face(processed_frame, face, worker.swap_face, 
                                                 f"GPU{worker.gpu_id}", track_id=i)
                    if newf is not None:
                        processed_frame = newf
        else:
            # 正常模式
            for i, face in enumerate(faces_sorted):
                if i >= len(worker.source_faces):
                    break
                
                src_face = worker.source_faces[i]
                emb = worker._get_embedding(face)
                
                if emb is None:
                    continue
                
                # 所有人脸都建立track
                worker.initial_frame_embeddings[i] = emb.copy()
                worker.track_embeddings[i] = {
                    "emb": emb,
                    "last_seen": frame_idx,
                    "original_emb": emb.copy(),
                    "history": []
                }
                worker.track_stability[i] = 1
                worker.track_source_map[i] = i
                
                if src_face is None:
                    # skip人脸：保存自身Face对象，执行自己换自己+透明化
                    worker.skip_face_self_faces[i] = face
                    
                    # 自己换自己，但会透明化贴回
                    newf = worker._safe_swap_face(processed_frame, face, face,
                                                 f"GPU{worker.gpu_id}", track_id=i)
                    if newf is not None:
                        processed_frame = newf
                    
                    if debug:
                        from core.checkpoint_manager import log_with_time
                        log_with_time("DEBUG", f"Track {i} 标记为SKIP（自己换自己+透明化）")
                else:
                    newf = worker._safe_swap_face(processed_frame, face, src_face, 
                                                 f"GPU{worker.gpu_id}", track_id=i)
                    if newf is not None:
                        processed_frame = newf
        
        worker.initial_mapping_done = True
        return processed_frame
    
    # 后续帧处理
    if worker._should_reset_tracking():
        worker._reset_tracking_with_reference(faces)
    
    worker._try_recover_lost_tracks(faces)
    
    # 匹配所有track
    sims = []
    for det_idx, face in enumerate(faces_sorted):
        emb = worker._get_embedding(face)
        if emb is None:
            continue
        
        for tid, t in worker.track_embeddings.items():
            if tid not in worker.track_source_map:
                continue
            
            avg_emb = worker._get_average_embedding(tid)
            is_match, sim = worker._adaptive_similarity_threshold(avg_emb, emb, tid)
            
            if is_match:
                sims.append((sim, det_idx, tid, emb))
    
    sims.sort(key=lambda x: x[0], reverse=True)
    used_tracks, used_faces = set(), set()
    new_embeddings = {}
    
    # 处理所有匹配
    for sim, det_idx, tid, emb in sims:
        if tid in used_tracks or det_idx in used_faces:
            continue
        
        current_face = faces_sorted[det_idx]
        
        if worker.swap_all_mode:
            # Swap-All模式
            src_type = worker.track_source_map.get(tid)
            if src_type is None:
                continue
            
            if src_type == "SKIP":
                # SKIP：自己换自己+透明化
                worker.skip_face_self_faces[tid] = current_face
                src_face = current_face  # 使用自己作为源脸
            else:
                # SWAP：换成源脸
                src_face = worker.swap_face
        else:
            # 正常模式
            src_idx = worker.track_source_map.get(tid)
            if src_idx is None:
                continue
            
            # 获取源人脸
            is_skip_track = tid in worker.skip_face_self_faces
            
            if is_skip_track:
                # SKIP：自己换自己+透明化
                worker.skip_face_self_faces[tid] = current_face
                src_face = current_face  # 使用自己作为源脸
            else:
                src_face = worker.source_faces[src_idx]
        
        if src_face is None:
            continue
        
        # 判断是否应该换脸
        should_swap = (
            sim > worker.sim_threshold + 0.06 or
            sim > worker.sim_threshold + 0.03 or
            (sim > worker.sim_threshold and worker._is_track_stable(tid, 1)) or
            (sim > worker.sim_threshold - 0.03 and worker._is_track_stable(tid, 2))
        )
        
        if should_swap:
            # 对于SKIP，传入自己作为源脸，_safe_swap_face会识别track_id并透明化
            newf = worker._safe_swap_face(processed_frame, current_face, 
                                         src_face, f"GPU{worker.gpu_id}", track_id=tid)
            if newf is not None:
                processed_frame = newf
                worker._increment_stability(tid)
                if debug:
                    from core.checkpoint_manager import log_with_time
                    if worker.swap_all_mode:
                        skip_info = " (SKIP)" if worker.track_source_map.get(tid) == "SKIP" else ""
                    else:
                        skip_info = " (SKIP)" if tid in worker.skip_face_self_faces else ""
                    log_with_time("DEBUG", f"帧{frame_idx} Track{tid}{skip_info} 替换成功 sim={sim:.3f}")
        elif sim > worker.sim_threshold - 0.03:
            worker._increment_stability(tid)
            # 更新SKIP track的缓存
            if worker.swap_all_mode and worker.track_source_map.get(tid) == "SKIP":
                worker.skip_face_self_faces[tid] = current_face
            elif tid in worker.skip_face_self_faces:
                worker.skip_face_self_faces[tid] = current_face
        
        # 更新track（SKIP和正常都需要）
        old_track = worker.track_embeddings.get(tid, {})
        new_embeddings[tid] = {
            "emb": emb,
            "last_seen": frame_idx,
            "original_emb": old_track.get("original_emb", emb),
            "history": old_track.get("history", [])
        }
        worker._update_embedding_history(tid, emb)
        
        used_tracks.add(tid)
        used_faces.add(det_idx)
    
    # Swap-All模式：处理未匹配的新人脸
    if worker.swap_all_mode:
        for det_idx, face in enumerate(faces_sorted):
            if det_idx in used_faces:
                continue
            
            emb = worker._get_embedding(face)
            if emb is None:
                continue
            
            # 创建新track，默认换成源脸
            new_tid = max(worker.track_embeddings.keys()) + 1 if worker.track_embeddings else 0
            
            worker.track_embeddings[new_tid] = {
                "emb": emb,
                "last_seen": frame_idx,
                "original_emb": emb.copy(),
                "history": []
            }
            worker.track_source_map[new_tid] = "SWAP"
            worker.track_stability[new_tid] = 1
            
            # 立即换脸
            newf = worker._safe_swap_face(processed_frame, face, worker.swap_face,
                                         f"GPU{worker.gpu_id} 新人脸", track_id=new_tid)
            if newf is not None:
                processed_frame = newf
                if debug:
                    from core.checkpoint_manager import log_with_time
                    log_with_time("DEBUG", f"帧{frame_idx} 新人脸 Track{new_tid} 换脸成功")
    
    # 保留未匹配但未过期的track
    for tid, t in worker.track_embeddings.items():
        if tid not in new_embeddings:
            if worker.max_age == -1 or (frame_idx - t["last_seen"]) <= worker.max_age:
                new_embeddings[tid] = t
                if tid in worker.track_stability and worker.track_stability[tid] > 1:
                    worker.track_stability[tid] = max(1, worker.track_stability[tid] - 1)
    
    worker.track_embeddings = new_embeddings
    worker.frame_idx = frame_idx
    worker._prune_tracks()
    
    return processed_frame


def _worker(gpu_id, source_faces, frame_paths, model_name='inswapper_128', 
            max_age=70, sim_threshold=0.16, reset_interval=60, 
            pixel_boost='256x256', auto_pixel_boost=False, 
            frame_resolution=None, debug=False):
    """单个worker处理函数"""
    worker = GPUWorker(gpu_id, source_faces, model_name=model_name, max_age=max_age, 
                       sim_threshold=sim_threshold, reset_interval=reset_interval, 
                       pixel_boost=pixel_boost, auto_pixel_boost=auto_pixel_boost,
                       frame_resolution=frame_resolution, debug=debug)
    
    if not worker.initialize_models():
        print(f"[ERROR] GPU {gpu_id} 模型初始化失败，跳过处理")
        return
    
    total_frames = len(frame_paths)
    processed = 0
    failed = 0
    
    for i, p in enumerate(frame_paths):
        success, msg = worker.process_frame(p)
        if success:
            processed += 1
        else:
            failed += 1
            if not debug:
                print(f"[ERROR] GPU {gpu_id} 处理 {p} 失败: {msg}")
        
        if debug and (i + 1) % 20 == 0:
            print(f"[DEBUG] GPU {gpu_id} 进度: {i+1}/{total_frames} (成功:{processed}, 失败:{failed})")
    
    print(f"[INFO] GPU {gpu_id} 完成处理: {total_frames} 帧 (成功:{processed}, 失败:{failed})")


def process_video(source_faces, frame_paths, use_multi_gpu=True, model_name='inswapper_128', 
                  max_age=70, sim_threshold=0.16, reset_interval=60, pixel_boost='256x256', 
                  auto_pixel_boost=False, debug=False):
    """处理视频（多worker模式，统一GPU/CPU处理）"""
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[INFO] 开始处理 {len(frame_paths)} 帧，使用模型: {model_name}")
    print(f"[INFO] 检测到GPU数量: {available_gpus}")
    print(f"[INFO] Pixel Boost设置: {pixel_boost}")
    if auto_pixel_boost:
        print(f"[INFO] 启用自动 Pixel Boost") 
    
    # 获取帧分辨率（从第一帧）
    frame_resolution = None
    if auto_pixel_boost and frame_paths:
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            frame_resolution = (w, h)

    if use_multi_gpu and available_gpus > 1:
        print(f"[INFO] 使用多GPU处理，逐帧轮询分配")
        
        workers = []
        for i in range(available_gpus):
            worker = GPUWorker(i, source_faces, model_name=model_name, max_age=max_age, 
                             sim_threshold=sim_threshold, reset_interval=reset_interval, 
                             pixel_boost=pixel_boost, auto_pixel_boost=auto_pixel_boost,
                             frame_resolution=frame_resolution, debug=debug)
            if worker.initialize_models():
                workers.append(worker)
            else:
                print(f"[ERROR] GPU {i} 初始化失败")
        
        if not workers:
            print("[ERROR] 没有可用的GPU worker")
            return
        
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = []
            for i, frame_path in enumerate(frame_paths):
                worker_idx = i % len(workers)
                future = executor.submit(workers[worker_idx].process_frame, frame_path)
                futures.append((future, worker_idx, frame_path))
            
            processed = 0
            failed = 0
            for future, worker_idx, frame_path in futures:
                try:
                    success, msg = future.result()
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        if not debug:
                            print(f"[ERROR] GPU {worker_idx} 处理 {frame_path} 失败: {msg}")
                except Exception as e:
                    failed += 1
                    print(f"[ERROR] GPU {worker_idx} 处理 {frame_path} 异常: {e}")
            
            print(f"[INFO] 多GPU处理完成: {len(frame_paths)} 帧 (成功:{processed}, 失败:{failed})")
            
    elif available_gpus > 0:
        print(f"[INFO] 使用单GPU处理，GPU ID: 0 (多GPU被禁用或仅有1个GPU)")
        _worker(0, source_faces, frame_paths, model_name, max_age, sim_threshold, reset_interval, pixel_boost, auto_pixel_boost, frame_resolution, debug)
    else:
        print(f"[INFO] 使用CPU处理")
        _worker(-1, source_faces, frame_paths, model_name, max_age, sim_threshold, reset_interval, pixel_boost, auto_pixel_boost, frame_resolution, debug)


def process_img(source_faces, target_path, model_name='inswapper_128', pixel_boost='256x256', auto_pixel_boost=False, debug=False):
    """处理单张图片"""
    gpu_id = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else -1

    worker = GPUWorker(gpu_id, source_faces, model_name=model_name, pixel_boost=pixel_boost, auto_pixel_boost=auto_pixel_boost, debug=debug)
    if not worker.initialize_models():
        print(f"[ERROR] 图片处理模型初始化失败")
        return False
    
    success, msg = worker.process_frame(target_path)
    if not success:
        print(f"[ERROR] 图片处理失败: {msg}")
    return success


def process_video_direct(source_faces, target_path, output_path, model_name='inswapper_128', 
                         max_age=70, sim_threshold=0.16, reset_interval=60, 
                         pixel_boost='256x256', skip_audio=False, 
                         auto_pixel_boost=False, debug=False, max_workers_per_gpu=4):
    """直接处理视频（已废弃，调用checkpoint版本）"""
    log_with_time("WARNING", "process_video_direct 已废弃，建议使用 process_video_direct_checkpoint")
    log_with_time("INFO", "将调用 process_video_direct_checkpoint 处理...")
    
    return process_video_direct_checkpoint(
        source_faces=source_faces,
        target_path=target_path,
        output_path=output_path,
        model_name=model_name,
        max_age=max_age,
        sim_threshold=sim_threshold,
        reset_interval=reset_interval,
        pixel_boost=pixel_boost,
        segment_frames=600,
        use_multi_gpu=True,
        skip_audio=skip_audio,
        auto_pixel_boost=auto_pixel_boost,
        debug=debug,
        max_workers_per_gpu=max_workers_per_gpu,
        start_frame=None,
        end_frame=None,
        track_frame=None,
        extract_only=False
    )
# ===== Checkpoint 处理主函数 =====

def process_video_direct_checkpoint(source_faces, target_path, output_path, 
                                    model_name='inswapper_128', max_age=70, 
                                    sim_threshold=0.16, reset_interval=60, 
                                    pixel_boost='256x256', segment_frames=600,
                                    use_multi_gpu=True, skip_audio=False, 
                                    auto_pixel_boost=False, debug=False,
                                    max_workers_per_gpu=4,
                                    start_frame=None, end_frame=None,
                                    track_frame=None, extract_only=False,
                                    encoder='libx264', crf=23, preset='medium',
                                    swap_all_mode=False, no_merge=False):  # 添加 no_merge 参数
    """直接处理视频,支持断点续传和部分处理"""
    from core.checkpoint_manager import CheckpointManager, log_with_time
    from core.processing_range import ProcessingRange
    import cv2
    import torch
    
    # 尝试导入psutil
    try:
        import psutil
    except ImportError:
        psutil = None
        log_with_time("WARNING", "psutil未安装，CPU内存分配将使用保守策略")
    
    # 初始化检查点管理器
    checkpoint = CheckpointManager(output_path, segment_frames, debug=debug)
    # 获取视频信息 - 使用精确帧数检测
    from core.utils import detect_fps, get_accurate_frame_count
    
    log_with_time("INFO", "="*60)
    log_with_time("INFO", "开始检测视频信息...")
    
    # 使用精确方法检测帧数
    total_frames, detection_method = get_accurate_frame_count(target_path)
    
    # 获取其他视频信息
    cap = cv2.VideoCapture(target_path)
    opencv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = detect_fps(target_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 对比不同方法的结果
    if detection_method != 'opencv':
        diff = abs(total_frames - opencv_frame_count)
        if diff > 0:
            log_with_time("INFO", f"帧数检测对比:")
            log_with_time("INFO", f"  {detection_method}: {total_frames} 帧")
            log_with_time("INFO", f"  opencv:          {opencv_frame_count} 帧")
            log_with_time("INFO", f"  差异:            {diff} 帧")
            
            if diff > 50:
                log_with_time("WARNING", 
                    f"检测差异较大({diff}帧)，可能是可变帧率(VFR)视频")
    log_with_time("INFO", f"最终使用: {total_frames} 帧 (方法: {detection_method})")
    log_with_time("INFO", f"帧率: {fps:.6f} fps")
    log_with_time("INFO", f"分辨率: {width}x{height}")
    log_with_time("INFO", "="*60)
    
    frame_resolution = (width, height)
    
    # 创建处理范围对象
    processing_range = ProcessingRange(
        total_frames, fps, start_frame, end_frame, None, None, track_frame
    )
    
    # 设置视频信息（会自动处理帧数不一致的情况）
    if extract_only:
        checkpoint.set_video_info(fps, width, height, processing_range.get_frame_count())
    else:
        checkpoint.set_video_info(fps, width, height, total_frames)

    # 提前检查是否已完成
    if checkpoint.is_fully_completed():
        log_with_time("INFO", "检测到任务已完成")
        # 检查输出文件是否存在
        if os.path.exists(output_path):
            log_with_time("INFO", f"输出文件已存在: {output_path}")
            
            # 验证输出文件的完整性
            try:
                cap = cv2.VideoCapture(output_path)
                output_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                expected_frames = checkpoint.checkpoint_data.get('total_frames', 0)
                
                if output_frames >= expected_frames * 0.95:  # 允许5%误差
                    log_with_time("INFO", f"输出文件验证通过 ({output_frames}/{expected_frames} 帧)")
                    checkpoint.cleanup()
                    return True
                else:
                    log_with_time("WARNING", 
                        f"输出文件帧数不足 ({output_frames}/{expected_frames})，将重新合并")
                    # 重置合并状态
                    checkpoint.checkpoint_data['merge_status'] = {
                        'segments_merged': False,
                        'audio_added': False
                    }
                    checkpoint._save_checkpoint()
            except Exception as e:
                log_with_time("WARNING", f"输出文件验证失败: {e}，将重新合并")
        
        # 如果输出文件不存在或验证失败，继续处理
        log_with_time("INFO", "输出文件缺失或不完整，开始合并...")
    
    # 获取视频信息
    from core.utils import detect_fps
    cap = cv2.VideoCapture(target_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = detect_fps(target_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 创建处理范围对象
    processing_range = ProcessingRange(
        total_frames, fps, start_frame, end_frame, None, None, track_frame
    )
    
    frame_resolution = (width, height)
    
    # 设置视频信息
    if extract_only:
        checkpoint.set_video_info(fps, width, height, processing_range.get_frame_count())
    else:
        checkpoint.set_video_info(fps, width, height, total_frames)
    
    checkpoint.set_encoder_config(encoder=encoder, crf=crf, preset=preset)
    log_with_time("INFO", f"编码器配置: {encoder}, CRF={crf}, Preset={preset}")

    # 获取需要处理的segment列表（传入processing_range）
    segments_to_process = checkpoint.get_segments_to_process(
        start_frame=processing_range.start_frame,
        end_frame=processing_range.end_frame
    )

    # 调试信息
    completed_frames, total = checkpoint.get_progress()
    log_with_time("INFO", f"当前进度: {completed_frames}/{total} 帧已完成")

    if not segments_to_process:
        log_with_time("INFO", "处理范围内所有segment已完成，开始合并...")
        if extract_only:
            return merge_and_finalize(checkpoint, target_path, skip_audio, 
                                     processing_range, extract_only)
        else:
            return merge_with_original(checkpoint, target_path, output_path, 
                                      processing_range, skip_audio)

    # 直接使用 segments_to_process，不再进行额外过滤
    resume_frame = segments_to_process[0][1]
    log_with_time("INFO", f"从帧 {resume_frame} 继续处理（处理范围: {processing_range.start_frame} - {processing_range.end_frame}）")
    
    # GPU设置
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log_with_time("INFO", f"检测到GPU数量: {available_gpus}")
    log_with_time("INFO", f"Pixel Boost设置: {pixel_boost}")
    if auto_pixel_boost:
        log_with_time("INFO", "启用自动 Pixel Boost")
    log_with_time("INFO", f"分段大小: {segment_frames} 帧")
    
    # 决定使用的GPU数量
    if use_multi_gpu and available_gpus > 1:
        log_with_time("INFO", f"使用多GPU模式处理 ({available_gpus} 个GPU)")
        gpu_ids = list(range(available_gpus))
    elif available_gpus > 0:
        log_with_time("INFO", "使用单GPU模式 (GPU 0) - 多worker")
        gpu_ids = [0]
    else:
        log_with_time("INFO", "未检测到GPU,使用CPU模式 - 多worker")
        gpu_ids = [-1]
    
    # 确定track参考帧
    track_reference_frame = 0
    if hasattr(processing_range, '_track_frame_specified') and processing_range._track_frame_specified:
        track_reference_frame = processing_range.track_frame
        log_with_time("INFO", f"使用指定的track帧: {track_reference_frame}")
    else:
        log_with_time("INFO", f"使用第一帧作为track参考: {track_reference_frame}")
    
    # 开始处理 - 多worker模式
    try:
        import threading
        import queue
        import time
        
        log_with_time("INFO", f"最大worker数限制: {max_workers_per_gpu}/GPU")
        
        # 智能分配worker
        balancer = GPUMemoryBalancer(gpu_ids, max_workers_per_gpu, debug)
        workers_config = balancer.balance_and_allocate()
        total_workers = sum(workers_config.values())
        
        # 任务队列和线程管理
        task_queue = queue.Queue(maxsize=100)
        result_queue = queue.Queue(maxsize=100)
        reading_done = threading.Event()
        oom_detected = threading.Event()
        
        # 统计
        gpu_stats = {gid: {'processed': 0, 'time': 0.0, 'workers': {}} for gid in gpu_ids}
        stats_lock = threading.Lock()
        
        # 帧读取线程
        def frame_reader_thread():
            reader_cap = cv2.VideoCapture(target_path)
            frame_idx = 0
            
            try:
                if resume_frame > 0:
                    log_with_time("INFO", f"需要续传，目标起始帧: {resume_frame}")
                    
                    # 策略1: 先尝试seek（快速但可能不准）
                    log_with_time("INFO", "尝试使用seek快速定位...")
                    reader_cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
                    actual_pos = int(reader_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    log_with_time("INFO", f"Seek结果: 请求帧{resume_frame}, 实际到达帧{actual_pos}")
                    
                    seek_diff = actual_pos - resume_frame
                    
                    if seek_diff == 0:
                        # Seek完全准确
                        log_with_time("INFO", "✓ Seek准确，直接从此位置开始")
                        frame_idx = actual_pos
                        
                    elif abs(seek_diff) <= 100:
                        # Seek偏差较小（±100帧以内）
                        if seek_diff > 0:
                            # Seek过头了，需要重新定位
                            log_with_time("WARNING", 
                                f"Seek过头{seek_diff}帧，从头跳过{resume_frame}帧以确保准确")
                            reader_cap.release()
                            reader_cap = cv2.VideoCapture(target_path)
                            
                            # 从头跳到目标位置
                            skip_start = time.time()
                            for i in range(resume_frame):
                                ret, _ = reader_cap.read()
                                if not ret:
                                    log_with_time("ERROR", f"跳帧失败于帧 {i}")
                                    reading_done.set()
                                    return
                                
                                if (i + 1) % 5000 == 0:
                                    elapsed = time.time() - skip_start
                                    speed = (i + 1) / elapsed
                                    remaining = (resume_frame - i - 1) / speed if speed > 0 else 0
                                    log_with_time("INFO", 
                                        f"跳帧进度: {i+1}/{resume_frame} ({(i+1)/resume_frame*100:.1f}%) "
                                        f"速度: {speed:.0f}帧/秒, 预计剩余: {remaining:.0f}秒")
                            
                            frame_idx = resume_frame
                            skip_elapsed = time.time() - skip_start
                            log_with_time("INFO", 
                                f"✓ 精确跳帧完成: {resume_frame}帧用时{skip_elapsed:.1f}秒")
                            
                        else:
                            # Seek不够远，只需补充跳过差额部分
                            log_with_time("INFO", 
                                f"Seek差{-seek_diff}帧，逐帧跳过差额部分...")
                            
                            frame_idx = actual_pos
                            skip_start = time.time()
                            
                            while frame_idx < resume_frame:
                                ret, _ = reader_cap.read()
                                if not ret:
                                    log_with_time("ERROR", f"补充跳帧失败于帧 {frame_idx}")
                                    reading_done.set()
                                    return
                                frame_idx += 1
                            
                            skip_elapsed = time.time() - skip_start
                            log_with_time("INFO", 
                                f"✓ 补充跳过{-seek_diff}帧完成，用时{skip_elapsed:.2f}秒")
                        
                    else:
                        # Seek偏差很大（>100帧），不可信，从头跳
                        log_with_time("WARNING", 
                            f"⚠️ Seek偏差过大({seek_diff}帧)，可能是VFR视频")
                        log_with_time("INFO", "为确保准确，将从头逐帧跳过...")
                        
                        reader_cap.release()
                        reader_cap = cv2.VideoCapture(target_path)
                        
                        skip_start = time.time()
                        frame_idx = 0
                        
                        while frame_idx < resume_frame:
                            ret, _ = reader_cap.read()
                            if not ret:
                                log_with_time("ERROR", f"跳帧失败于帧 {frame_idx}")
                                reading_done.set()
                                return
                            
                            frame_idx += 1
                            
                            if frame_idx % 5000 == 0:
                                elapsed = time.time() - skip_start
                                speed = frame_idx / elapsed
                                remaining = (resume_frame - frame_idx) / speed if speed > 0 else 0
                                log_with_time("INFO", 
                                    f"跳帧进度: {frame_idx}/{resume_frame} ({frame_idx/resume_frame*100:.1f}%) "
                                    f"速度: {speed:.0f}帧/秒, 预计剩余: {remaining:.0f}秒")
                        
                        skip_elapsed = time.time() - skip_start
                        log_with_time("INFO", 
                            f"✓ 完整跳帧完成: {resume_frame}帧用时{skip_elapsed:.1f}秒 "
                            f"(速度: {resume_frame/skip_elapsed:.0f}帧/秒)")
                    
                    log_with_time("INFO", f"✓ 准确定位完成，从帧 {frame_idx} 开始处理")
                else:
                    log_with_time("INFO", "从帧 0 开始读取")
                
                # 现在开始实际读取和处理
                frames_read = 0
                read_start_frame = frame_idx
                
                while frame_idx < processing_range.end_frame and not oom_detected.is_set():
                    ret, frame = reader_cap.read()
                    if not ret:
                        log_with_time("WARNING", f"读取失败于帧 {frame_idx}")
                        if frames_read > 0:
                            log_with_time("INFO", 
                                f"实际读取: 从帧{read_start_frame}到帧{frame_idx-1}, 共{frames_read}帧")
                        break
                    
                    task_queue.put((frame_idx, frame.copy()))
                    frame_idx += 1
                    frames_read += 1
                    
                    if frames_read % 100 == 0:
                        log_with_time("INFO", f"读取: {frames_read}帧 (当前帧号: {frame_idx})")
                
                if oom_detected.is_set():
                    log_with_time("ERROR", "检测到OOM,停止读取")
                else:
                    log_with_time("INFO", 
                        f"读取完成: 从帧{read_start_frame}到帧{frame_idx-1}, 共{frames_read}帧")
                
                reading_done.set()
                
                # 等待队列消费
                wait_count = 0
                while not task_queue.empty() and wait_count < 600:
                    time.sleep(0.5)
                    wait_count += 1
                
                # 发送结束信号
                for _ in range(total_workers):
                    task_queue.put((None, None))
            
            except Exception as e:
                log_with_time("ERROR", f"读取错误: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
            finally:
                reader_cap.release()
                reading_done.set()
        
        # GPU工作线程
        def gpu_worker_thread(gpu_id, worker_id):
            """GPU worker线程"""
            thread_name = f"GPU{gpu_id}W{worker_id}"
            processed = 0
            failed = 0
            total_time = 0.0
            
            try:
                log_with_time("INFO", f"{thread_name} 开始初始化...")
                
                # 创建独立的worker实例
                worker = GPUWorker(
                    gpu_id, source_faces,
                    model_name=model_name,
                    max_age=max_age,
                    sim_threshold=sim_threshold,
                    reset_interval=reset_interval,
                    pixel_boost=pixel_boost,
                    auto_pixel_boost=auto_pixel_boost,
                    frame_resolution=frame_resolution,
                    debug=debug,
                    swap_all_mode=swap_all_mode
                )
                
                if not worker.initialize_models():
                    log_with_time("ERROR", f"{thread_name} 初始化失败")
                    oom_detected.set()
                    return
                
                log_with_time("INFO", f"{thread_name} 初始化成功")
                
                # 建立track映射（使用track_reference_frame）
                cap_init = cv2.VideoCapture(target_path)
                cap_init.set(cv2.CAP_PROP_POS_FRAMES, track_reference_frame)
                ret, first_frame = cap_init.read()
                cap_init.release()
                
                if ret and first_frame is not None:
                    faces = worker.models["analyser"].get(first_frame)
                    if faces:
                        faces_sorted = sorted(faces, key=lambda x: x.bbox[0])
                        
                        if swap_all_mode:
                            # Swap-All模式的初始化
                            for i, face in enumerate(faces_sorted):
                                emb = worker._get_embedding(face)
                                
                                if emb is None:
                                    continue
                                
                                worker.initial_frame_embeddings[i] = emb.copy()
                                worker.track_embeddings[i] = {
                                    "emb": emb,
                                    "last_seen": track_reference_frame,
                                    "original_emb": emb.copy(),
                                    "history": []
                                }
                                worker.track_stability[i] = 1
                                
                                if i in worker.skip_positions:
                                    worker.track_source_map[i] = "SKIP"
                                    worker.skip_face_self_faces[i] = face
                                    if debug:
                                        log_with_time("DEBUG", f"{thread_name} Track {i} 设为SKIP（位置{i}）")
                                else:
                                    worker.track_source_map[i] = "SWAP"
                                    if debug:
                                        log_with_time("DEBUG", f"{thread_name} Track {i} 设为SWAP")
                            
                        else:
                            # 正常模式的初始化
                            for i, face in enumerate(faces_sorted):
                                if i >= len(worker.source_faces):
                                    break
                                
                                src_face = worker.source_faces[i]
                                emb = worker._get_embedding(face)
                                
                                if emb is None:
                                    continue
                                
                                worker.initial_frame_embeddings[i] = emb.copy()
                                worker.track_embeddings[i] = {
                                    "emb": emb,
                                    "last_seen": track_reference_frame,
                                    "original_emb": emb.copy(),
                                    "history": []
                                }
                                worker.track_stability[i] = 1
                                worker.track_source_map[i] = i
                                
                                if src_face is None:
                                    worker.skip_face_self_faces[i] = face
                        
                        worker.initial_mapping_done = True
                
                # 处理循环
                while not oom_detected.is_set():
                    try:
                        frame_idx, frame = task_queue.get(timeout=2.0)
                        
                        if frame_idx is None:
                            break
                        
                        start_time = time.time()
                        processed_frame = None
                        
                        try:
                            faces = worker.models["analyser"].get(frame)
                            
                            if not faces:
                                worker.frame_idx = frame_idx
                                worker._prune_tracks()
                                processed_frame = frame
                            else:
                                processed_frame = _process_frame_with_worker(
                                    worker, frame, faces, frame_idx, debug
                                )
                            
                            if processed_frame is None:
                                log_with_time("WARNING", f"{thread_name} 帧 {frame_idx} 处理返回None,使用原帧")
                                processed_frame = frame
                                failed += 1
                            
                        except Exception as e:
                            log_with_time("ERROR", f"{thread_name} 帧 {frame_idx} 处理异常: {str(e)[:200]}")
                            if debug:
                                import traceback
                                traceback.print_exc()
                            
                            # 使用原帧继续
                            processed_frame = frame
                            failed += 1
                            
                            # 只有OOM才终止
                            if "out of memory" in str(e).lower():
                                log_with_time("ERROR", f"{thread_name} 检测到OOM，停止处理")
                                oom_detected.set()
                        
                        # 关键修改：无论处理成功还是失败，都必须放入结果队列
                        if processed_frame is not None:
                            try:
                                result_queue.put((frame_idx, processed_frame), timeout=5.0)
                                processed += 1
                            except queue.Full:
                                log_with_time("ERROR", f"{thread_name} 结果队列已满，帧 {frame_idx} 可能丢失")
                                # 即使队列满了，也要尽力放入
                                result_queue.put((frame_idx, processed_frame))
                                processed += 1
                        else:
                            log_with_time("ERROR", f"{thread_name} 帧 {frame_idx} 处理后为None，使用原帧")
                            result_queue.put((frame_idx, frame))
                            failed += 1
                        
                        frame_time = time.time() - start_time
                        total_time += frame_time
                        
                        # 更新统计
                        with stats_lock:
                            gpu_stats[gpu_id]['processed'] += 1
                            gpu_stats[gpu_id]['time'] += frame_time
                            gpu_stats[gpu_id]['workers'][worker_id] = {
                                'processed': processed,
                                'failed': failed,
                                'time': total_time
                            }
                        
                        if processed % 50 == 0 and processed > 0:
                            avg = (total_time / processed) * 1000
                            fail_info = f" (失败{failed})" if failed > 0 else ""
                            log_with_time("INFO", f"{thread_name}: {processed}帧{fail_info} | {avg:.0f}ms/帧")
                    
                    except queue.Empty:
                        if reading_done.is_set():
                            break
                        continue
                
                if oom_detected.is_set():
                    log_with_time("WARNING", f"{thread_name} 因OOM停止")
                else:
                    fail_info = f" (失败{failed})" if failed > 0 else ""
                    log_with_time("INFO", f"{thread_name} 完成: {processed}帧{fail_info}")
            
            except Exception as e:
                log_with_time("ERROR", f"{thread_name} 严重错误: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                if "out of memory" in str(e).lower():
                    oom_detected.set()
        
        # 启动所有线程
        reader_thread = threading.Thread(target=frame_reader_thread, daemon=True)
        reader_thread.start()
        
        # 为每个GPU启动多个worker线程
        all_threads = []
        for gpu_id in gpu_ids:
            num_workers_for_gpu = workers_config[gpu_id]
            for worker_id in range(num_workers_for_gpu):
                thread = threading.Thread(
                    target=gpu_worker_thread,
                    args=(gpu_id, worker_id),
                    daemon=True
                )
                thread.start()
                all_threads.append(thread)
            
            log_with_time("INFO", f"GPU{gpu_id} 启动了{num_workers_for_gpu}个worker")
        
        # 主线程写入逻辑
        log_with_time("INFO", "主线程开始写入...")
        time.sleep(min(10, 2 * total_workers))  # 等待worker初始化

        video_info = checkpoint.checkpoint_data['video_info']

        # 使用缓冲区处理乱序帧
        current_segment_idx = segments_to_process[0][0]
        segment_writer = None
        segment_start_frame = segments_to_process[0][1]
        segment_end_frame = segments_to_process[0][2]
        actual_segment_start = segment_start_frame  # 记录实际开始写入的第一帧

        results_buffer = {}
        next_write_idx = segment_start_frame
        consecutive_timeouts = 0
        last_report = time.time()
        skipped_frames = set()
        last_progress_frame = next_write_idx  # 用于检测是否有进度

        log_with_time("INFO", f"开始写入，从帧 {segment_start_frame} 到 {processing_range.end_frame}")

        while next_write_idx < processing_range.end_frame:
            if oom_detected.is_set():
                if result_queue.empty() and next_write_idx not in results_buffer:
                    log_with_time("WARNING", f"OOM导致处理中断于帧 {next_write_idx}")
                    break
            
            # 创建或切换segment writer
            if segment_writer is None or next_write_idx >= segment_end_frame:
                if segment_writer is not None:
                    # 关闭当前segment
                    actual_written = segment_writer.release()
                    
                    # 使用实际写入的帧范围
                    actual_frames_written = next_write_idx - actual_segment_start
                    
                    log_with_time("INFO", 
                        f"Segment {current_segment_idx}: "
                        f"计划帧{segment_start_frame}-{segment_end_frame}, "
                        f"实际写入帧{actual_segment_start}-{next_write_idx-1}, "
                        f"写入{actual_frames_written}帧, writer统计{actual_written}帧")
                    
                    checkpoint.mark_segment_completed(current_segment_idx, actual_frames_written)
                
                # 查找下一个segment
                found_next = False
                for seg_idx, seg_start, seg_end in segments_to_process:
                    if seg_start == next_write_idx:
                        current_segment_idx = seg_idx
                        segment_start_frame = seg_start
                        segment_end_frame = seg_end
                        actual_segment_start = next_write_idx
                        segment_writer = SegmentWriter(checkpoint, seg_idx, video_info)
                        log_with_time("INFO", f"创建 segment_{seg_idx} (帧 {seg_start} - {seg_end})")
                        found_next = True
                        break
                
                if not found_next:
                    log_with_time("ERROR", f"找不到帧 {next_write_idx} 对应的segment")
                    break
            
            # 获取帧
            processed_frame = None
            
            if next_write_idx in results_buffer:
                processed_frame = results_buffer.pop(next_write_idx)
                consecutive_timeouts = 0
            else:
                try:
                    frame_idx, frame = result_queue.get(timeout=0.1)
                    
                    if frame_idx == next_write_idx:
                        processed_frame = frame
                        consecutive_timeouts = 0
                    else:
                        results_buffer[frame_idx] = frame
                        
                        if len(results_buffer) > 100:
                            buffered_frames = sorted(results_buffer.keys())
                            log_with_time("WARNING", 
                                f"Buffer积压 {len(results_buffer)} 帧, "
                                f"当前等待 {next_write_idx}, "
                                f"buffer范围: {buffered_frames[0]}-{buffered_frames[-1]}")
                        
                        continue
                
                except queue.Empty:
                    consecutive_timeouts += 1
                    
                    # 动态计算超时阈值
                    if oom_detected.is_set():
                        max_timeout = 100  # OOM时快速失败
                    elif reading_done.is_set() and all(not t.is_alive() for t in all_threads):
                        max_timeout = 50   # 所有worker完成时快速失败
                    else:
                        max_timeout = 1000  # 正常处理时给更多时间
                    
                    if consecutive_timeouts > max_timeout:
                        log_with_time("ERROR", f"超时等待帧 {next_write_idx}")
                        log_with_time("ERROR", f"连续超时次数: {consecutive_timeouts}")
                        
                        all_workers_done = reading_done.is_set() and all(not t.is_alive() for t in all_threads)
                        
                        if all_workers_done:
                            log_with_time("ERROR", "所有worker已完成，但仍缺少帧！")
                            
                            if results_buffer:
                                buffered_frames = sorted(results_buffer.keys())
                                log_with_time("WARNING", f"Buffer中的帧: {buffered_frames[:20]}")
                                
                                if buffered_frames[0] > next_write_idx:
                                    log_with_time("ERROR", 
                                        f"检测到丢帧: 帧{next_write_idx}丢失, "
                                        f"下一个可用帧: {buffered_frames[0]}")
                                    
                                    # 尝试从原视频恢复
                                    try:
                                        recovery_cap = cv2.VideoCapture(target_path)
                                        recovery_cap.set(cv2.CAP_PROP_POS_FRAMES, next_write_idx)
                                        ret, recovery_frame = recovery_cap.read()
                                        recovery_cap.release()
                                        
                                        if ret and recovery_frame is not None:
                                            log_with_time("WARNING", f"从原视频恢复帧 {next_write_idx}")
                                            processed_frame = recovery_frame
                                            skipped_frames.add(next_write_idx)
                                            consecutive_timeouts = 0  # 重置计数
                                        else:
                                            log_with_time("ERROR", f"无法从原视频恢复帧 {next_write_idx}")
                                            log_with_time("ERROR", "停止处理，保存当前进度")
                                            break
                                    except Exception as e:
                                        log_with_time("ERROR", f"恢复帧失败: {e}")
                                        log_with_time("ERROR", "停止处理，保存当前进度")
                                        break
                                else:
                                    log_with_time("ERROR", "无法确定丢失的帧，停止处理")
                                    break
                            else:
                                log_with_time("ERROR", "Buffer为空且所有worker已完成，停止处理")
                                break
                        else:
                            # Worker还在运行，但长时间没有进度
                            if consecutive_timeouts % 100 == 0:
                                log_with_time("WARNING", 
                                    f"等待帧 {next_write_idx} 已超时 {consecutive_timeouts} 次，"
                                    f"reader_done={reading_done.is_set()}, "
                                    f"active_workers={sum(1 for t in all_threads if t.is_alive())}")
                            continue
                    else:
                        continue
            
            # 写入帧
            if processed_frame is not None:
                if processed_frame.shape[:2] != (height, width):
                    processed_frame = cv2.resize(processed_frame, (width, height))
                
                segment_writer.expect_frame(next_write_idx)
                
                if segment_writer.write(processed_frame, next_write_idx):
                    next_write_idx += 1
                    last_progress_frame = next_write_idx  # 更新进度
                else:
                    log_with_time("ERROR", f"写入帧 {next_write_idx} 失败!")
                    break
            
            # 进度报告
            now = time.time()
            if next_write_idx % 100 == 0 or (now - last_report) > 10:
                progress = (next_write_idx - segment_start_frame) / (processing_range.end_frame - segment_start_frame) * 100
                
                with stats_lock:
                    summary = []
                    for gid in gpu_ids:
                        proc = gpu_stats[gid]['processed']
                        t = gpu_stats[gid]['time']
                        avg = (t / proc * 1000) if proc > 0 else 0
                        summary.append(f"GPU{gid}:{proc}({avg:.0f}ms)")
                
                log_with_time("INFO", 
                    f"写入: {next_write_idx}/{processing_range.end_frame} ({progress:.1f}%) | "
                    f"Buffer: {len(results_buffer)} | {' '.join(summary)}")
                last_report = now

        # 关闭最后的segment
        if segment_writer is not None:
            actual_written = segment_writer.release()
            actual_frames_written = next_write_idx - actual_segment_start
            
            log_with_time("INFO", 
                f"最后 segment {current_segment_idx}: "
                f"实际写入帧{actual_segment_start}-{next_write_idx-1}, "
                f"写入{actual_frames_written}帧")
            
            checkpoint.mark_segment_completed(current_segment_idx, actual_frames_written)

        if skipped_frames:
            log_with_time("WARNING", f"共有 {len(skipped_frames)} 帧从原视频恢复: {sorted(skipped_frames)[:20]}")
        
        # 等待线程
        log_with_time("INFO", "等待工作线程结束...")
        reader_thread.join(timeout=10)
        for t in all_threads:
            t.join(timeout=5)
        
        # 最终验证
        completed_frames, total = checkpoint.get_progress()
        actual_frames_written = next_write_idx - resume_frame
        
        log_with_time("INFO", 
            f"写入完成: 本次写入 {actual_frames_written} 帧, "
            f"总进度 {completed_frames}/{total}")
        
        # 统计
        log_with_time("INFO", "\n=== GPU/CPU统计 ===")
        total_processed = 0
        for gid in gpu_ids:
            stats = gpu_stats[gid]
            total_proc = stats['processed']
            total_processed += total_proc
            total_t = stats['time']
            
            if total_proc > 0:
                avg = (total_t / total_proc) * 1000
                target_frames = processing_range.end_frame - resume_frame
                pct = (total_proc / target_frames) * 100 if target_frames > 0 else 0
            else:
                avg = 0.0
                pct = 0.0
            
            log_with_time("INFO", f"GPU {gid}: {total_proc}帧({pct:.1f}%) | 平均{avg:.0f}ms/帧")
        
        # 检查是否成功完成
        if next_write_idx < processing_range.end_frame:
            log_with_time("WARNING", 
                f"处理未完成: 已写入 {next_write_idx} 帧，目标 {processing_range.end_frame} 帧")
            log_with_time("INFO", "进度已保存，可以重新运行继续处理")
            return False
        
        # OOM情况下的处理
        if oom_detected.is_set():
            completed_frames = checkpoint.get_progress()[0]
            log_with_time("WARNING", 
                f"因OOM中断,已处理{completed_frames}帧(目标{processing_range.end_frame}帧)")
            log_with_time("INFO", 
                f"进度已保存,建议用更少worker恢复: --max-workers-per-gpu {max(1, max_workers_per_gpu-1)}")
            return False

        
        
        # 新增：检查是否需要跳过合并
        if no_merge:
            log_with_time("INFO", "=" * 60)
            log_with_time("INFO", "所有帧处理完成")
            log_with_time("INFO", f"--no-merge: 跳过合并阶段，保留临时文件")
            log_with_time("INFO", f"临时文件位置: {checkpoint.temp_dir}")
            log_with_time("INFO", f"Segment文件: {checkpoint.temp_dir}/segment_*.mp4")
            log_with_time("INFO", "=" * 60)
            log_with_time("INFO", "如需合并，请去掉 --no-merge 参数重新运行")
            return True
        
        # 合并
        log_with_time("INFO", "所有帧处理完成，开始合并分段...")
        
        if extract_only:
            return merge_and_finalize(checkpoint, target_path, skip_audio, 
                                     processing_range, extract_only)
        else:
            return merge_with_original(checkpoint, target_path, output_path, 
                                      processing_range, skip_audio)
    
    except Exception as e:
        log_with_time("ERROR", f"处理过程出错: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        log_with_time("INFO", f"进度已保存到: {checkpoint.temp_dir}")
        log_with_time("INFO", "可使用相同命令恢复处理")
        return False


# ===== 合并函数 =====

def merge_and_finalize(checkpoint, target_path, skip_audio, 
                      processing_range, extract_only):
    """合并处理的片段（不包含原视频其他部分）"""
    from core.checkpoint_manager import log_with_time
    import os
    import cv2
    
    log_with_time("INFO", "合并处理的片段...")
    
    # 从原视频获取准确的fps
    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        video_info = checkpoint.checkpoint_data.get('video_info', {})
        fps = video_info.get('fps', 30.0)
    
    audio_start_time = processing_range.start_frame / fps if fps > 0 else 0.0
    
    log_with_time("INFO", f"音频对齐: start_frame={processing_range.start_frame}, fps={fps:.2f}, 起始时间={audio_start_time:.2f}s")
    
    merge_success = checkpoint.merge_segments(target_path, skip_audio, audio_start_time)
    
    if merge_success:
        log_with_time("INFO", f"片段提取成功: {checkpoint.output_path}")
        checkpoint.cleanup()
        return True
    else:
        log_with_time("ERROR", "片段合并失败")
        return False


def merge_with_original(checkpoint, target_path, output_path, 
                       processing_range, skip_audio):
    """将处理的片段合并回原视频"""
    from core.checkpoint_manager import log_with_time
    import os
    import cv2
    
    log_with_time("INFO", "开始合并到原视频...")
    
    # 判断是否需要拼接原视频的前后部分
    need_concat = (processing_range.start_frame > 0 or 
                   processing_range.end_frame < processing_range.total_frames)
    
    if not need_concat:
        # 情况1：处理了整个视频，直接合并segments
        log_with_time("INFO", "处理了完整视频，直接合并segments")
        checkpoint.output_path = output_path
        success = checkpoint.merge_segments(target_path, skip_audio)
        
        if success:
            log_with_time("INFO", f"完整视频处理成功: {output_path}")
            checkpoint.cleanup()
            return True
        else:
            log_with_time("ERROR", "视频合并失败")
            return False
    
    # 情况2：只处理了部分，需要拼接原视频
    log_with_time("INFO", "处理了部分视频，需要拼接原视频前后部分")
    
    # 步骤1：合并处理的segments为临时文件
    processed_clip = os.path.join(checkpoint.temp_dir, 'processed_clip.mp4')
    checkpoint.output_path = processed_clip
    
    # 获取准确的fps和音频起始时间
    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        video_info = checkpoint.checkpoint_data.get('video_info', {})
        fps = video_info.get('fps', 30.0)
    
    audio_start_time = processing_range.start_frame / fps if fps > 0 else 0.0
    
    merge_success = checkpoint.merge_segments(target_path, skip_audio=True, audio_start_time=audio_start_time)
    if not merge_success:
        log_with_time("ERROR", "处理片段合并失败")
        return False
    
    # 步骤2：提取原视频的前后部分
    start_time = processing_range.start_frame / fps
    end_time = processing_range.end_frame / fps
    
    before_clip = os.path.join(checkpoint.temp_dir, 'before.mp4')
    after_clip = os.path.join(checkpoint.temp_dir, 'after.mp4')
    
    # 提取前部分
    if processing_range.start_frame > 0:
        log_with_time("INFO", f"提取原视频前部分: 0 - {start_time:.2f}s")
        cmd = f'ffmpeg -i "{target_path}" -t {start_time:.6f} -c copy -y "{before_clip}" -hide_banner -loglevel error'
        os.system(cmd)
    
    # 提取后部分
    if processing_range.end_frame < processing_range.total_frames:
        log_with_time("INFO", f"提取原视频后部分: {end_time:.2f}s - 结束")
        cmd = f'ffmpeg -i "{target_path}" -ss {end_time:.6f} -c copy -y "{after_clip}" -hide_banner -loglevel error'
        os.system(cmd)
    
    # 步骤3：拼接视频片段
    log_with_time("INFO", "拼接视频片段...")
    concat_file = os.path.join(checkpoint.temp_dir, 'concat_final.txt')
    
    with open(concat_file, 'w') as f:
        if os.path.exists(before_clip):
            f.write(f"file '{os.path.basename(before_clip)}'\n")
        f.write(f"file '{os.path.basename(processed_clip)}'\n")
        if os.path.exists(after_clip):
            f.write(f"file '{os.path.basename(after_clip)}'\n")
    
    temp_output = os.path.join(checkpoint.temp_dir, 'final_no_audio.mp4')
    cmd = f'cd "{checkpoint.temp_dir}" && ffmpeg -f concat -safe 0 -i concat_final.txt -c copy -y final_no_audio.mp4 -hide_banner -loglevel error'
    result = os.system(cmd)
    
    if result != 0 or not os.path.exists(temp_output):
        log_with_time("ERROR", "视频拼接失败")
        return False
    
    # 步骤4：添加音频
    if not skip_audio:
        log_with_time("INFO", "添加音频...")
        cmd = f'ffmpeg -i "{temp_output}" -i "{target_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y "{output_path}" -hide_banner -loglevel error'
        result = os.system(cmd)
    else:
        import shutil
        shutil.copy(temp_output, output_path)
        result = 0
    
    if result == 0 and os.path.exists(output_path):
        log_with_time("INFO", f"视频合并成功: {output_path}")
        checkpoint.cleanup()
        return True
    else:
        log_with_time("ERROR", "最终视频生成失败")
        return False
