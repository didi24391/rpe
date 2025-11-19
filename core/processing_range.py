# core/processing_range.py
class ProcessingRange:
    """视频处理范围管理"""
    
    def __init__(self, total_frames, fps, 
                 start_frame=None, end_frame=None,
                 start_time=None, end_time=None,
                 track_frame=None):
        self.total_frames = total_frames
        self.fps = fps
        
        # 解析时间为帧号
        self.start_frame = self._parse_frame(start_frame, start_time, fps, 0)
        self.end_frame = self._parse_frame(end_frame, end_time, fps, total_frames)
        
        # 边界检查
        self.start_frame = max(0, min(self.start_frame, total_frames))
        self.end_frame = max(self.start_frame, min(self.end_frame, total_frames))
        
        # Track frame逻辑
        if track_frame is not None and track_frame >= 0:
            self.track_frame = max(0, min(track_frame, total_frames - 1))
            self._track_frame_specified = True
            print(f"[INFO] ✓ 使用指定的 track_frame={self.track_frame}")
        else:
            self.track_frame = self.start_frame
            self._track_frame_specified = False
            print(f"[INFO] 未指定 track_frame，使用处理范围第一帧={self.track_frame}")
        
        self._print_info()
    
    def _parse_frame(self, frame, time_str, fps, default):
        """解析帧号或时间字符串"""
        if frame is not None:
            return frame
        if time_str is not None:
            from core.utils import parse_time_to_seconds, time_to_frame
            seconds = parse_time_to_seconds(time_str)
            return time_to_frame(seconds, fps)
        return default
    
    def _print_info(self):
        """打印处理范围信息"""
        from core.utils import frame_to_time
        
        print(f"[INFO] ===== 处理范围 =====")
        print(f"[INFO] 视频总帧数: {self.total_frames} (帧率: {self.fps:.2f} fps)")
        print(f"[INFO] 处理范围: 帧 {self.start_frame} - {self.end_frame}")
        print(f"[INFO]   时间: {frame_to_time(self.start_frame, self.fps)} - {frame_to_time(self.end_frame, self.fps)}")
        print(f"[INFO]   共 {self.end_frame - self.start_frame} 帧")
        
        track_type = "[用户指定]" if self._track_frame_specified else "[默认=第一帧]"
        print(f"[INFO] Track参考帧: 帧 {self.track_frame} (时间: {frame_to_time(self.track_frame, self.fps)}) {track_type}")
        print(f"[INFO] ========================")
    
    def should_process_frame(self, frame_idx):
        """判断是否应该处理该帧"""
        return self.start_frame <= frame_idx < self.end_frame
    
    def get_frame_count(self):
        """获取需要处理的帧数"""
        return self.end_frame - self.start_frame
