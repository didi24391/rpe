# core/utils.py
import os
import shutil
import cv2

def is_img(path):
    """检查是否为图片文件"""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def path_safe(p):
    """路径安全处理(当前为直通)"""
    return p

def detect_fps(input_path):
    """检测视频帧率，返回浮点数(保留完整精度)"""
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
        out = os.popen(cmd).read().strip()
        
        if "/" in out:
            num, den = out.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(out)
        
        # 合理性检查
        if fps <= 0 or fps > 240:
            print(f"[WARNING] 检测到异常帧率 {fps:.6f}，使用默认值 30.0")
            return 30.0
        
        print(f"[INFO] 检测到视频帧率: {fps:.6f} fps")
        return fps
    except Exception as e:
        print(f"[WARNING] 帧率检测失败: {e}，使用默认值 30.0")
        return 30.0

def get_video_resolution(input_path):
    """获取视频分辨率"""
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "{input_path}"'
        out = os.popen(cmd).read().strip()
        if 'x' in out:
            width, height = out.split('x')
            return int(width), int(height)
    except:
        pass
    return 1920, 1080

def extract_frames(input_path, output_dir):
    """提取视频帧"""
    cmd = f'ffmpeg -i "{input_path}" "{os.path.join(output_dir, "%04d.png")}" -hide_banner -loglevel error'
    os.system(cmd)

def add_audio(frames_dir, target_path, keep_frames, output_file, skip_audio=False):
    """音频合成逻辑"""
    fps = detect_fps(target_path)
    frames_glob = os.path.join(frames_dir, "%04d.png")
    
    # 检查原视频是否有音频
    audio_check_cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 "{target_path}" 2>/dev/null'
    has_audio = os.popen(audio_check_cmd).read().strip()
    
    if skip_audio or not has_audio:
        print(f"[INFO] 创建无音频视频")
        cmd = f'ffmpeg -framerate {fps:.6f} -i "{frames_glob}" -c:v libx264 -crf 23 -pix_fmt yuv420p -r {fps:.6f} -y "{output_file}" -hide_banner -loglevel error'
    else:
        print(f"[INFO] 检测到音频，正在合并...")
        cmd = f'ffmpeg -framerate {fps:.6f} -i "{frames_glob}" -i "{target_path}" -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac -map 0:v:0 -map 1:a:0 -shortest -r {fps:.6f} -y "{output_file}" -hide_banner -loglevel error'
    
    result = os.system(cmd)
    
    if result != 0 and has_audio and not skip_audio:
        print("[WARNING] 音频合并失败，创建无音频版本")
        cmd = f'ffmpeg -framerate {fps:.6f} -i "{frames_glob}" -c:v libx264 -crf 23 -pix_fmt yuv420p -r {fps:.6f} -y "{output_file}" -hide_banner -loglevel error'
        os.system(cmd)
    
    # 清理帧目录
    if not keep_frames:
        try:
            shutil.rmtree(frames_dir)
            print("[INFO] 已清理临时帧文件")
        except Exception as e:
            print(f"[WARNING] 清理帧文件失败: {e}")

def parse_time_to_seconds(time_str):
    """解析时间字符串为秒数。支持: "90", "1:30", "0:01:30" """
    if time_str is None:
        return None
    
    try:
        return float(time_str)
    except ValueError:
        pass
    
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    raise ValueError(f"无效的时间格式: {time_str}")

def time_to_frame(seconds, fps):
    """将时间转换为帧号"""
    return int(seconds * fps) if seconds is not None else None

def frame_to_time(frame_num, fps):
    """将帧号转换为时间字符串 (HH:MM:SS)"""
    if frame_num is None:
        return None
    
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"
def get_accurate_frame_count(video_path):
    """获取精确的视频帧数（优先使用ffprobe）
    
    Returns:
        (frame_count, method) - 帧数和检测方法
    """
    import subprocess
    
    # 方法1: ffprobe的nb_read_packets（最准确，逐包计数）
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-count_packets', '-show_entries', 'stream=nb_read_packets', 
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            ffprobe_count = int(result.stdout.strip())
            print(f"[INFO] ffprobe精确计数: {ffprobe_count} 帧")
            return ffprobe_count, 'ffprobe_packets'
    except Exception as e:
        print(f"[WARNING] ffprobe计数失败: {e}")
    
    # 方法2: ffprobe的nb_frames（次优）
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'stream=nb_frames', 
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            ffprobe_frames = int(result.stdout.strip())
            print(f"[INFO] ffprobe nb_frames: {ffprobe_frames} 帧")
            return ffprobe_frames, 'ffprobe_frames'
    except Exception as e:
        print(f"[WARNING] ffprobe nb_frames失败: {e}")
    
    # 方法3: OpenCV（不准确，尤其是VFR视频）
    try:
        cap = cv2.VideoCapture(video_path)
        opencv_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[WARNING] 使用OpenCV检测（可能不准确）: {opencv_count} 帧")
        return opencv_count, 'opencv'
    except Exception as e:
        print(f"[ERROR] OpenCV检测失败: {e}")
        return 0, 'failed'
