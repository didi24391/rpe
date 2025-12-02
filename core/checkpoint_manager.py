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
    
    def __init__(self, output_path: str, segment_frames: int = 600, debug: bool = False):
        self.output_path = output_path
        self.segment_frames = segment_frames
        self.debug = debug  # 添加这一行
        
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
        """验证和修复checkpoint数据 - 优化版（随机抽样验证）"""
        import random
        
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
        
        # 统一键类型为整数（修复类型不一致问题）
        completed_segments = {}
        for k, v in self.checkpoint_data.get('completed_segments', {}).items():
            completed_segments[int(k)] = v
        
        if not completed_segments:
            return
        
        total_segments = len(completed_segments)
        sample_size = max(1, min(int(total_segments * 0.1), 50))
        
        log_with_time("INFO", f"验证已完成的segment（抽样检查 {sample_size}/{total_segments} 个）...")
        
        # 随机选择要验证的segment
        segment_ids = list(completed_segments.keys())
        sample_ids = random.sample(segment_ids, sample_size)
        
        valid_segments = completed_segments.copy()
        corrupted_count = 0
        
        for seg_idx in sample_ids:
            seg_idx = int(seg_idx)
            seg_path = self.get_segment_path(seg_idx)
            recorded_frames = completed_segments[seg_idx]
            
            if not os.path.exists(seg_path):
                log_with_time("WARNING", f"  segment_{seg_idx}: 文件不存在，已移除")
                valid_segments.pop(seg_idx, None)
                corrupted_count += 1
                continue
            
            actual_frames = self._verify_segment_frames(seg_idx)
            if actual_frames == 0:
                log_with_time("WARNING", f"  segment_{seg_idx}: 损坏，已移除")
                try:
                    os.remove(seg_path)
                except:
                    pass
                valid_segments.pop(seg_idx, None)
                corrupted_count += 1
                continue
            
            # 更新为实际帧数
            if actual_frames != recorded_frames:
                log_with_time("INFO", f"  segment_{seg_idx}: 更新帧数 {recorded_frames} → {actual_frames}")
                valid_segments[seg_idx] = actual_frames
            else:
                log_with_time("INFO", f"  segment_{seg_idx}: {actual_frames} 帧 ✓")
        
        self.checkpoint_data['completed_segments'] = valid_segments
        
        # 显示验证结果
        if corrupted_count > 0:
            log_with_time("WARNING", f"发现 {corrupted_count} 个损坏的segment，已移除")
        else:
            log_with_time("INFO", f"抽样验证通过，所有检查的segment完整")
        
        # 显示实际完成的总帧数
        if valid_segments:
            total_completed_frames = sum(valid_segments.values())
            sorted_segs = sorted([int(k) for k in valid_segments.keys()])
            if len(sorted_segs) <= 20:
                log_with_time("INFO", f"已完成segment: {sorted_segs}")
            else:
                log_with_time("INFO", f"已完成segment: {sorted_segs[:10]}...{sorted_segs[-10:]} (共{len(sorted_segs)}个)")
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
        """验证segment的实际帧数 - 优化版（使用ffprobe快速检测）"""
        seg_path = self.get_segment_path(segment_idx)
        if not os.path.exists(seg_path):
            return 0
        
        try:
            # 方法1: 使用ffprobe快速获取帧数（推荐）
            cmd = f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "{seg_path}"'
            result = os.popen(cmd).read().strip()
            
            if result and result.isdigit():
                return int(result)
            
            # 方法2: 备用 - 使用OpenCV的CAP_PROP_FRAME_COUNT（快但可能不准）
            cap = cv2.VideoCapture(seg_path)
            if not cap.isOpened():
                return 0
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 如果返回值合理，直接使用
            if frame_count > 0:
                return frame_count
            
            return 0
            
        except Exception as e:
            log_with_time("WARNING", f"快速验证segment {segment_idx} 失败: {e}，尝试逐帧验证")
            
            # 方法3: 最后备用 - 逐帧读取（慢但最准确）
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
            except:
                return 0
    
    def get_segment_path(self, segment_idx: int) -> str:
        """获取分段视频路径"""
        return os.path.join(self.temp_dir, f'segment_{segment_idx:04d}.mp4')
    
    def is_segment_completed(self, segment_idx: int) -> bool:
        """检查分段是否已完成"""
        completed = self.checkpoint_data.get('completed_segments', {})
        # 统一使用整数类型
        return int(segment_idx) in [int(k) for k in completed.keys()]
    
    def mark_segment_completed(self, segment_idx: int, frames_written: int):
        """标记分段完成"""
        completed = self.checkpoint_data.get('completed_segments', {})
        # 统一使用整数键
        completed[int(segment_idx)] = frames_written
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
    
    def get_resume_frame(self, start_frame=None, end_frame=None) -> int:
        """获取恢复处理的起始帧 - 简化版
        
        Args:
            start_frame: 处理范围起始帧（可选）
            end_frame: 处理范围结束帧（可选）
        
        Returns:
            下一个需要处理的帧号
        """
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.checkpoint_data.get('total_frames', 0)
        
        completed = self.checkpoint_data.get('completed_segments', {})
        
        if not completed:
            return start_frame
        
        # 统一键类型
        completed_dict = {}
        for k, v in completed.items():
            completed_dict[int(k)] = v
        
        # 从start_frame开始，找到第一个未完成的segment
        current_frame = start_frame
        seg_idx = start_frame // self.segment_frames
        
        while current_frame < end_frame:
            if seg_idx not in completed_dict:
                # 找到第一个未完成的segment
                log_with_time("INFO", f"从segment {seg_idx}（帧 {current_frame}）继续处理")
                return current_frame
            
            # 这个segment已完成，移动到下一个
            current_frame += self.segment_frames
            seg_idx += 1
        
        # 所有segment都完成了
        log_with_time("INFO", "所有segment已完成")
        return end_frame
    
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
    
    def get_segments_to_process(self, start_frame=None, end_frame=None) -> List[Tuple[int, int, int]]:
        """获取需要处理的segment列表（修复版 - 不假设连续性）
        
        Args:
            start_frame: 处理范围起始帧（可选）
            end_frame: 处理范围结束帧（可选）
        
        Returns: List of (segment_idx, start_frame, end_frame)
        """
        total_frames = self.checkpoint_data.get('total_frames', 0)
        if total_frames == 0:
            return []
        
        # 如果指定了处理范围，使用处理范围
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames
        
        # 统一键类型为整数
        completed_dict = {}
        for k, v in self.checkpoint_data.get('completed_segments', {}).items():
            completed_dict[int(k)] = v
        
        # 计算处理范围内应该有的所有segment
        segments_to_process = []
        current_frame = start_frame
        seg_idx = start_frame // self.segment_frames
        
        while current_frame < end_frame:
            seg_end = min(current_frame + self.segment_frames, end_frame)
            
            # 检查这个segment是否已完成
            if seg_idx in completed_dict:
                # 已完成，验证文件
                seg_path = self.get_segment_path(seg_idx)
                if os.path.exists(seg_path):
                    file_size = os.path.getsize(seg_path)
                    if file_size >= 1024:
                        # 文件存在且看起来正常，跳过
                        if self.debug:
                            log_with_time("DEBUG", f"Segment {seg_idx} 已完成且文件完整，跳过")
                    else:
                        log_with_time("WARNING", f"Segment {seg_idx} 文件异常({file_size}B)，重新处理")
                        self.remove_segment_from_checkpoint(seg_idx)
                        segments_to_process.append((seg_idx, current_frame, seg_end))
                else:
                    log_with_time("WARNING", f"Segment {seg_idx} 文件丢失，重新处理")
                    self.remove_segment_from_checkpoint(seg_idx)
                    segments_to_process.append((seg_idx, current_frame, seg_end))
            else:
                # 未完成，需要处理
                segments_to_process.append((seg_idx, current_frame, seg_end))
            
            current_frame = seg_end
            seg_idx += 1
        
        if completed_dict:
            completed_count = len([s for s in range(start_frame // self.segment_frames, 
                                                    (end_frame + self.segment_frames - 1) // self.segment_frames)
                                  if s in completed_dict])
            total_in_range = (end_frame - start_frame + self.segment_frames - 1) // self.segment_frames
            log_with_time("INFO", f"处理范围内: 已完成 {completed_count}/{total_in_range} 个segment")
        
        if segments_to_process:
            log_with_time("INFO", f"需要处理 {len(segments_to_process)} 个segment:")
            for seg_idx, start, end in segments_to_process[:5]:
                log_with_time("INFO", f"  segment_{seg_idx}: 帧 {start} - {end}")
            if len(segments_to_process) > 5:
                log_with_time("INFO", f"  ... 还有 {len(segments_to_process) - 5} 个")
        else:
            log_with_time("INFO", "处理范围内所有segment已完成")
        
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
        """合并所有分段视频 - 优化版（抽样验证 + 时长检查）"""
        import random
        
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
        
        # 优化验证：随机抽样10%的segment，最少1个，最多不超过50个
        completed_count = len(completed)
        sample_size = max(1, min(int(completed_count * 0.1), 50))
        
        log_with_time("INFO", f"验证segment完整性（抽样检查 {sample_size}/{completed_count} 个）...")
        
        # 随机选择要验证的segment
        segment_ids = sorted(completed.keys())
        sample_ids = random.sample(segment_ids, sample_size)
        
        valid_segments = []
        total_frames = 0
        corrupted_count = 0
        
        # 验证抽样的segment
        for seg_idx in sample_ids:
            seg_path = self.get_segment_path(seg_idx)
            if not os.path.exists(seg_path):
                log_with_time("ERROR", f"  segment_{seg_idx}: 文件缺失")
                corrupted_count += 1
                continue
            
            expected_frames = completed[seg_idx]
            actual_frames = self._verify_segment_frames(seg_idx)
            
            if actual_frames != expected_frames:
                log_with_time("WARNING", 
                    f"  segment_{seg_idx}: 帧数不匹配 (记录:{expected_frames}, 实际:{actual_frames})")
            else:
                log_with_time("INFO", f"  segment_{seg_idx}: {actual_frames} 帧 ✓")
        
        # 如果抽样发现问题，提示用户
        if corrupted_count > 0:
            log_with_time("WARNING", f"抽样发现 {corrupted_count} 个segment有问题，建议检查")
        else:
            log_with_time("INFO", "抽样验证通过")
        
        # 所有segment都添加到合并列表（即使没被抽样验证）
        for seg_idx in segment_ids:
            valid_segments.append(seg_idx)
            total_frames += completed[seg_idx]
        
        if not valid_segments:
            log_with_time("ERROR", "没有有效的segment可以合并")
            return False
        
        log_with_time("INFO", f"准备合并 {len(valid_segments)} 个segment，预计总计: {total_frames} 帧")
        
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
            
            log_with_time("INFO", f"合并完成")
            
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
        
        # 步骤3: 验证最终视频时长
        log_with_time("INFO", "验证最终视频时长...")
        duration_check = self._verify_final_video_duration(target_path)
        
        if not duration_check:
            log_with_time("WARNING", "时长验证未通过，保留临时文件用于检查")
            self.checkpoint_data['keep_temp_for_review'] = True
            self._save_checkpoint()
        
        return True
    
    def _verify_final_video_duration(self, original_path: str) -> bool:
        """验证最终视频时长 - 智能策略
        
        策略：
        1. 对于短视频（<5分钟），允许1秒误差
        2. 对于中等视频（5-30分钟），允许0.1%误差（最少1秒）
        3. 对于长视频（>30分钟），允许0.1%误差（最多2秒）
        
        Returns:
            True: 时长验证通过
            False: 时长差异过大，需要保留临时文件
        """
        try:
            # 获取原视频时长
            cmd_original = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{original_path}"'
            original_duration = float(os.popen(cmd_original).read().strip())
            
            # 获取最终视频时长
            cmd_final = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{self.output_path}"'
            final_duration = float(os.popen(cmd_final).read().strip())
            
            duration_diff = abs(original_duration - final_duration)
            
            # 计算允许的误差范围
            if original_duration < 300:  # <5分钟
                allowed_diff = 1.0
                reason = "短视频（<5分钟）"
            elif original_duration < 1800:  # 5-30分钟
                allowed_diff = max(1.0, original_duration * 0.001)  # 0.1%，最少1秒
                reason = "中等视频（5-30分钟）"
            else:  # >30分钟
                allowed_diff = min(2.0, original_duration * 0.001)  # 0.1%，最多2秒
                reason = "长视频（>30分钟）"
            
            log_with_time("INFO", f"原视频时长: {original_duration:.2f}s ({self._format_duration(original_duration)})")
            log_with_time("INFO", f"最终视频时长: {final_duration:.2f}s ({self._format_duration(final_duration)})")
            log_with_time("INFO", f"时长差异: {duration_diff:.2f}s")
            log_with_time("INFO", f"允许误差: {allowed_diff:.2f}s ({reason})")
            
            if duration_diff > allowed_diff:
                diff_percent = (duration_diff / original_duration) * 100
                log_with_time("WARNING", 
                    f"时长差异超过允许范围 (差异: {duration_diff:.2f}s / {diff_percent:.2f}%, "
                    f"允许: {allowed_diff:.2f}s)")
                return False
            else:
                log_with_time("INFO", f"时长验证通过 ✓ (差异在允许范围内)")
                return True
                
        except Exception as e:
            log_with_time("WARNING", f"时长验证失败: {e}")
            # 验证失败时保守处理，保留临时文件
            return False
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长为易读格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def cleanup(self):
        """清理临时文件 - 优化版（检查时长验证结果）"""
        try:
            import shutil
            
            if not os.path.exists(self.temp_dir):
                return
            
            # 调试模式保留临时文件
            if os.getenv('DEBUG_KEEP_TEMP') == '1':
                log_with_time("INFO", f"调试模式：保留临时文件在 {self.temp_dir}")
                return
            
            # 检查是否需要保留临时文件用于审查
            if self.checkpoint_data.get('keep_temp_for_review', False):
                log_with_time("WARNING", "检测到视频时长差异超过2秒，保留临时文件用于检查")
                log_with_time("INFO", f"临时文件位置: {self.temp_dir}")
                log_with_time("INFO", "如需手动清理，请删除该文件夹")
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
