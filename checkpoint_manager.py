# core/checkpoint_manager.py
import os
import json
import cv2
import subprocess
from typing import Dict, List, Optional, Tuple
from datetime import datetime

def log_with_time(level, msg):
    """添加UTC时间戳的日志"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp} UTC] [{level}] {msg}")

class CheckpointManager:
    """断点续传管理器"""
    
    def __init__(self, output_path: str, segment_frames: int = 600):
        self.output_path = output_path
        self.segment_frames = segment_frames
        
        # 创建临时目录
        self.temp_dir = output_path.replace('.mp4', '_segments')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 检查点文件路径
        self.checkpoint_file = os.path.join(self.temp_dir, 'checkpoint.json')
        
        # 加载或初始化检查点数据
        self.checkpoint_data = self._load_checkpoint()
        
        # 验证和修复checkpoint数据
        self._validate_and_repair_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """加载检查点数据"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    log_with_time("INFO", f"加载检查点: 已完成 {len(data.get('completed_segments', []))} 个segment")
                    return data
            except Exception as e:
                log_with_time("WARNING", f"检查点文件损坏: {e}，重新开始")
        
        return {
            'completed_segments': {},  # {seg_idx: frame_count}
            'total_frames': 0,
            'video_info': {},
            'segment_frames': self.segment_frames,
            'merge_status': {
                'segments_merged': False,
                'audio_added': False
            }
        }
    
    def _save_checkpoint(self):
        """保存检查点数据"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
        except Exception as e:
            log_with_time("WARNING", f"保存检查点失败: {e}")
    
    def _validate_and_repair_checkpoint(self):
        """验证和修复checkpoint数据"""
        # 兼容旧版本的list格式
        completed = self.checkpoint_data.get('completed_segments', {})
        if isinstance(completed, list):
            log_with_time("INFO", "检测到旧版checkpoint格式，正在转换...")
            new_completed = {}
            for seg_idx in completed:
                frame_count = self._verify_segment_frames(seg_idx)
                if frame_count > 0:
                    new_completed[seg_idx] = frame_count
            self.checkpoint_data['completed_segments'] = new_completed
            self._save_checkpoint()
        
        # 验证所有已完成的segment
        log_with_time("INFO", "验证已完成的segment...")
        completed_segments = self.checkpoint_data.get('completed_segments', {})
        valid_segments = {}
        
        for seg_idx, recorded_frames in list(completed_segments.items()):
            seg_idx = int(seg_idx)
            seg_path = self.get_segment_path(seg_idx)
            
            if not os.path.exists(seg_path):
                log_with_time("WARNING", f"  segment_{seg_idx}: 文件不存在，已移除")
                continue
            
            actual_frames = self._verify_segment_frames(seg_idx)
            if actual_frames == 0:
                log_with_time("WARNING", f"  segment_{seg_idx}: 损坏，已移除")
                os.remove(seg_path)
                continue
            
            # 更新为实际帧数
            valid_segments[seg_idx] = actual_frames
            if actual_frames != recorded_frames:
                log_with_time("INFO", f"  segment_{seg_idx}: 更新帧数 {recorded_frames} → {actual_frames}")
            else:
                log_with_time("INFO", f"  segment_{seg_idx}: {actual_frames} 帧 ✓")
        
        self.checkpoint_data['completed_segments'] = valid_segments
        
        # 显示实际完成的总帧数
        if valid_segments:
            total_completed_frames = sum(valid_segments.values())
            sorted_segs = sorted([int(k) for k in valid_segments.keys()])
            log_with_time("INFO", f"已完成segment: {sorted_segs}")
            log_with_time("INFO", f"实际完成总帧数: {total_completed_frames}")
        
        self._save_checkpoint()
        
        # 更新segment_frames参数
        old_segment_frames = self.checkpoint_data.get('segment_frames')
        if old_segment_frames != self.segment_frames:
            log_with_time("INFO", f"segment_frames参数已更新: {old_segment_frames} → {self.segment_frames}")
            log_with_time("INFO", "将使用新的segment_frames划分剩余帧，已完成的segment保持不变")
            self.checkpoint_data['segment_frames'] = self.segment_frames
            self._save_checkpoint()
    
    def _verify_segment_frames(self, segment_idx: int) -> int:
        """验证segment的实际帧数（逐帧读取）"""
        seg_path = self.get_segment_path(segment_idx)
        if not os.path.exists(seg_path):
            return 0
        
        try:
            cap = cv2.VideoCapture(seg_path)
            if not cap.isOpened():
                return 0
            
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            cap.release()
            return frame_count
        except Exception as e:
            log_with_time("WARNING", f"验证segment {segment_idx} 失败: {e}")
            return 0
    
    def get_segment_path(self, segment_idx: int) -> str:
        """获取分段视频路径"""
        return os.path.join(self.temp_dir, f'segment_{segment_idx:04d}.mp4')
    
    def is_segment_completed(self, segment_idx: int) -> bool:
        """检查分段是否已完成"""
        completed = self.checkpoint_data.get('completed_segments', {})
        return segment_idx in completed
    
    def mark_segment_completed(self, segment_idx: int, frames_written: int):
        """标记分段完成"""
        completed = self.checkpoint_data.get('completed_segments', {})
        completed[segment_idx] = frames_written
        self.checkpoint_data['completed_segments'] = completed
        self._save_checkpoint()
        
        log_with_time("INFO", 
            f"Segment {segment_idx} 标记完成 ({frames_written} 帧)，"
            f"总进度: {len(completed)}/{self._get_total_segments()} segments")
    
    def _get_total_segments(self) -> int:
        """计算总segment数量"""
        total_frames = self.checkpoint_data.get('total_frames', 0)
        if total_frames == 0:
            return 0
        return (total_frames + self.segment_frames - 1) // self.segment_frames
    
    def get_resume_frame(self) -> int:
        """获取恢复处理的起始帧（使用实际完成的帧数）"""
        completed = self.checkpoint_data.get('completed_segments', {})
        
        if not completed:
            return 0
        
        # 按segment索引排序，累加实际帧数
        total_completed_frames = 0
        sorted_segments = sorted([int(k) for k in completed.keys()])
        
        # 找到第一个缺失的segment
        expected_seg_idx = 0
        for seg_idx in sorted_segments:
            if seg_idx != expected_seg_idx:
                # 找到了gap，从这里恢复
                log_with_time("INFO", f"找到缺失的segment {expected_seg_idx}，从帧 {total_completed_frames} 继续")
                return total_completed_frames
            
            # 累加这个segment的实际帧数
            total_completed_frames += completed[seg_idx]
            expected_seg_idx += 1
        
        # 所有已记录的segment都是连续的，从下一个segment开始
        log_with_time("INFO", f"已完成 {len(sorted_segments)} 个连续segment，从帧 {total_completed_frames} 继续")
        return total_completed_frames
    
    def get_next_segment_index(self) -> int:
        """获取下一个segment索引（找到第一个缺失的）"""
        completed = self.checkpoint_data.get('completed_segments', {})
        
        if not completed:
            return 0
        
        # 找到第一个缺失的segment索引
        sorted_segments = sorted([int(k) for k in completed.keys()])
        
        # 检查是否有gap
        for i, seg_idx in enumerate(sorted_segments):
            if seg_idx != i:
                # 发现gap，返回缺失的索引
                return i
        
        # 没有gap，返回下一个
        return len(sorted_segments)
    
    def get_segments_to_process(self) -> List[Tuple[int, int, int]]:
        """获取需要处理的segment列表
        Returns: List of (segment_idx, start_frame, end_frame)
        """
        total_frames = self.checkpoint_data.get('total_frames', 0)
        if total_frames == 0:
            return []
        
        completed = self.checkpoint_data.get('completed_segments', {})
        segments_to_process = []
        
        # 使用实际完成的帧数计算起始位置
        sorted_segments = sorted([int(k) for k in completed.keys()])
        
        current_frame = 0
        expected_seg_idx = 0
        
        # 先跳过所有已完成的连续segment
        for seg_idx in sorted_segments:
            if seg_idx != expected_seg_idx:
                # 发现gap，停止跳过
                break
            
            # 累加实际帧数
            current_frame += completed[seg_idx]
            expected_seg_idx += 1
        
        # 从当前位置开始，按新的segment_frames划分剩余帧
        seg_idx = expected_seg_idx
        while current_frame < total_frames:
            end_frame = min(current_frame + self.segment_frames, total_frames)
            
            # 检查这个segment是否已完成
            if seg_idx not in completed:
                segments_to_process.append((seg_idx, current_frame, end_frame))
            else:
                # 已完成但验证文件
                seg_path = self.get_segment_path(seg_idx)
                if not os.path.exists(seg_path):
                    log_with_time("WARNING", f"Segment {seg_idx} 标记为完成但文件不存在，将重新生成")
                    self.remove_segment_from_checkpoint(seg_idx)
                    segments_to_process.append((seg_idx, current_frame, end_frame))
                else:
                    actual_frames = self._verify_segment_frames(seg_idx)
                    expected_frames = completed[seg_idx]
                    if actual_frames == 0 or abs(actual_frames - expected_frames) > 2:
                        log_with_time("WARNING", 
                            f"Segment {seg_idx} 损坏 (预期{expected_frames}, 实际{actual_frames})，将重新生成")
                        self.remove_segment_from_checkpoint(seg_idx)
                        segments_to_process.append((seg_idx, current_frame, end_frame))
            
            current_frame = end_frame
            seg_idx += 1
        
        return segments_to_process
    
    def get_progress(self) -> Tuple[int, int]:
        """获取进度信息（已完成帧数, 总帧数）"""
        total_frames = self.checkpoint_data.get('total_frames', 0)
        completed = self.checkpoint_data.get('completed_segments', {})
        
        # 计算已完成的帧数
        completed_frames = sum(completed.values())
        
        return completed_frames, total_frames
    
    def set_video_info(self, fps: float, width: int, height: int, total_frames: int):
        """设置视频信息"""
        self.checkpoint_data['video_info'] = {
            'fps': fps,
            'width': width,
            'height': height
        }
        self.checkpoint_data['total_frames'] = total_frames
        self._save_checkpoint()
    
    def set_encoder_config(self, encoder='libx264', crf=23, preset='medium'):
        """设置编码器配置"""
        if 'video_info' not in self.checkpoint_data:
            self.checkpoint_data['video_info'] = {}
        
        self.checkpoint_data['video_info']['encoder'] = encoder
        self.checkpoint_data['video_info']['crf'] = crf
        self.checkpoint_data['video_info']['preset'] = preset
        self._save_checkpoint()
    
    def merge_segments(self, target_path: str, skip_audio: bool = False, audio_start_time: float = None) -> bool:
        """合并所有分段视频"""
        completed = self.checkpoint_data.get('completed_segments', {})
        
        if not completed:
            log_with_time("ERROR", "没有可合并的分段")
            return False
        
        # 检查是否所有segment都完成
        total_segments = self._get_total_segments()
        if len(completed) < total_segments:
            missing = []
            for seg_idx in range(total_segments):
                if seg_idx not in completed:
                    missing.append(seg_idx)
            log_with_time("WARNING", f"仍有 {len(missing)} 个segment未完成: {missing[:10]}")
            log_with_time("WARNING", "将只合并已完成的segment")
        
        # 验证所有segment文件
        log_with_time("INFO", "验证segment完整性...")
        valid_segments = []
        total_frames = 0
        
        for seg_idx in sorted(completed.keys()):
            seg_path = self.get_segment_path(seg_idx)
            if not os.path.exists(seg_path):
                log_with_time("ERROR", f"  segment_{seg_idx}: 文件缺失")
                continue
            
            expected_frames = completed[seg_idx]
            actual_frames = self._verify_segment_frames(seg_idx)
            
            if actual_frames != expected_frames:
                log_with_time("WARNING", 
                    f"  segment_{seg_idx}: 帧数不匹配 (记录:{expected_frames}, 实际:{actual_frames})")
            else:
                log_with_time("INFO", f"  segment_{seg_idx}: {actual_frames} 帧 ✓")
            
            valid_segments.append(seg_idx)
            total_frames += actual_frames
        
        if not valid_segments:
            log_with_time("ERROR", "没有有效的segment可以合并")
            return False
        
        log_with_time("INFO", f"验证完成，总计: {total_frames} 帧")
        
        # 获取视频信息
        video_info = self.checkpoint_data.get('video_info', {})
        fps = video_info.get('fps', 30.0)
        
        # 定义文件路径
        temp_merged = os.path.join(self.temp_dir, 'merged_no_audio.mp4')
        merge_status = self.checkpoint_data.get('merge_status', {})
        
        # 步骤1: 合并分段
        if not merge_status.get('segments_merged', False):
            log_with_time("INFO", f"开始合并 {len(valid_segments)} 个分段...")
            
            # 创建文件列表
            concat_file = os.path.join(self.temp_dir, 'concat_list.txt')
            with open(concat_file, 'w') as f:
                for seg_idx in valid_segments:
                    f.write(f"file '{os.path.basename(self.get_segment_path(seg_idx))}'\n")
            
            # 合并segment
            cmd = f'cd "{self.temp_dir}" && ffmpeg -f concat -safe 0 -i concat_list.txt -c copy -y merged_no_audio.mp4 -hide_banner -loglevel error'
            result = os.system(cmd)
            
            if result != 0 or not os.path.exists(temp_merged):
                log_with_time("ERROR", "分段合并失败")
                return False
            
            # 验证合并后的帧数
            merged_frames = self._verify_segment_frames(-1)
            if merged_frames == 0:
                cap = cv2.VideoCapture(temp_merged)
                merged_frames = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    merged_frames += 1
                cap.release()
            
            log_with_time("INFO", f"合并完成，验证帧数: {merged_frames}")
            
            if abs(merged_frames - total_frames) > 2:
                log_with_time("WARNING", 
                    f"合并后帧数不匹配：预期:{total_frames}, 实际:{merged_frames}, 差:{abs(merged_frames-total_frames)}")
            
            self.checkpoint_data['merge_status']['segments_merged'] = True
            self._save_checkpoint()
        else:
            log_with_time("INFO", "检测到已合并的分段")
            if not os.path.exists(temp_merged):
                log_with_time("ERROR", "merged_no_audio.mp4 丢失，重新合并")
                self.checkpoint_data['merge_status']['segments_merged'] = False
                self._save_checkpoint()
                return self.merge_segments(target_path, skip_audio, audio_start_time)
        
        # 步骤2: 添加音频
        if not merge_status.get('audio_added', False):
            if skip_audio:
                log_with_time("INFO", "--skip-audio: 直接重新编码...")
                cmd = f'ffmpeg -i "{temp_merged}" -c:v copy -y "{self.output_path}" -hide_banner -loglevel error'
                result = os.system(cmd)
            else:
                # 检测音频
                try:
                    audio_check = subprocess.run(
                        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
                         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', target_path],
                        capture_output=True, text=True, timeout=10
                    )
                    has_audio = audio_check.returncode == 0 and 'audio' in audio_check.stdout.lower()
                except Exception as e:
                    log_with_time("WARNING", f"音频检测失败: {e}, 假设有音频")
                    has_audio = True
                
                if has_audio:
                    if audio_start_time is not None and audio_start_time > 0:
                        log_with_time("INFO", f"添加音频轨道 (起始时间: {audio_start_time:.2f}s)...")
                        cmd = f'ffmpeg -i "{temp_merged}" -ss {audio_start_time:.6f} -i "{target_path}" -c:v copy -c:a aac -b:a 128k -map 0:v:0 -map 1:a:0 -shortest -y "{self.output_path}" -hide_banner -loglevel error'
                    else:
                        log_with_time("INFO", "添加音频轨道 (从头开始)...")
                        cmd = f'ffmpeg -i "{temp_merged}" -i "{target_path}" -c:v copy -c:a aac -b:a 128k -map 0:v:0 -map 1:a:0 -shortest -y "{self.output_path}" -hide_banner -loglevel error'
                    result = os.system(cmd)
                else:
                    log_with_time("INFO", "原视频无音频，直接输出...")
                    cmd = f'ffmpeg -i "{temp_merged}" -c:v copy -y "{self.output_path}" -hide_banner -loglevel error'
                    result = os.system(cmd)
            
            if result != 0 or not os.path.exists(self.output_path):
                log_with_time("ERROR", "最终视频生成失败")
                return False
            
            self.checkpoint_data['merge_status']['audio_added'] = True
            self._save_checkpoint()
            log_with_time("INFO", f"视频合成成功: {self.output_path}")
        else:
            log_with_time("INFO", "最终视频已存在")
            if not os.path.exists(self.output_path):
                log_with_time("ERROR", "最终文件丢失，重新生成")
                self.checkpoint_data['merge_status']['audio_added'] = False
                self._save_checkpoint()
                return self.merge_segments(target_path, skip_audio, audio_start_time)
        
        return True
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            
            if not os.path.exists(self.temp_dir):
                return
            
            # 调试模式保留临时文件
            if os.getenv('DEBUG_KEEP_TEMP') == '1':
                log_with_time("INFO", f"调试模式：保留临时文件在 {self.temp_dir}")
                return
            
            # 删除临时目录
            shutil.rmtree(self.temp_dir)
            log_with_time("INFO", "已清理临时文件")
            
        except Exception as e:
            log_with_time("WARNING", f"清理失败: {e}")
    
    def is_fully_completed(self) -> bool:
        """检查是否完全完成"""
        merge_status = self.checkpoint_data.get('merge_status', {})
        return (merge_status.get('segments_merged', False) and 
                merge_status.get('audio_added', False) and
                os.path.exists(self.output_path))
    
    def remove_segment_from_checkpoint(self, segment_idx: int):
        """从checkpoint中移除segment"""
        completed = self.checkpoint_data.get('completed_segments', {})
        if segment_idx in completed:
            del completed[segment_idx]
            self._save_checkpoint()
