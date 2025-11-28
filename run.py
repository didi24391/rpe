#!/usr/bin/env python3
# run.py
import argparse
import os
import shutil
import cv2

from core.processor import process_video, process_img, process_video_direct_checkpoint
from core.utils import is_img, detect_fps, add_audio, extract_frames, path_safe
from core.config import get_face, ModelManager


def load_source_faces(face_paths, use_hybrid, swap_all_mode=False, skip_positions=None):
    """加载源人脸（支持swap-all模式）"""
    from core.config import get_face, get_face_hybrid, get_face_model, is_hybrid_mode
    
    # swap-all模式：只加载一个源脸用于所有替换
    if swap_all_mode:
        if not face_paths or len(face_paths) == 0:
            print(f"[ERROR] swap-all模式需要提供至少一个源人脸")
            return None
        
        # 加载唯一的源脸
        swap_face_path = face_paths[0]
        img = cv2.imread(swap_face_path)
        if img is None:
            print(f"[ERROR] 无法读取源图像: {swap_face_path}")
            return None
        
        if is_hybrid_mode() and get_face_model().lower() in ("antelopev2", "buffalo_sc", "buffalo_l"):
            print(f"[INFO] 使用混合检测模式加载源人脸 ({get_face_model()}+buffalo_l)")
            swap_face = get_face_hybrid(img)
        else:
            swap_face = get_face(img)
        
        if swap_face is None:
            print(f"[ERROR] 源图像中未检测到人脸: {swap_face_path}")
            return None
        
        print(f"[INFO] swap-all模式: 已加载源人脸 ({swap_face_path})")
        print(f"[INFO] 跳过位置: {skip_positions if skip_positions else '无'}")
        
        return {
            'mode': 'swap_all',
            'swap_face': swap_face,
            'skip_positions': skip_positions or []
        }
    
    # 原有的正常模式
    source_faces = []
    for i, f in enumerate(face_paths):
        if f.lower() == "skip":
            source_faces.append(None)
            print(f"[INFO] 源人脸 {i}: skip")
            continue

        img = cv2.imread(f)
        if img is None:
            print(f"[ERROR] 无法读取源图像: {f}")
            return None

        if is_hybrid_mode() and get_face_model().lower() in ("antelopev2", "buffalo_sc", "buffalo_l"):
            print(f"[INFO] 使用混合检测模式加载源人脸 ({get_face_model()}+buffalo_l)")
            face = get_face_hybrid(img)
        else:
            face = get_face(img)

        if face is None:
            print(f"[ERROR] 源图像中未检测到人脸: {f}")
            return None
        
        source_faces.append(face)
        print(f"[INFO] 已加载源人脸 {i} ({f})")
    
    return source_faces


def verify_track_frame(target_path, track_frame, source_faces, debug=False, swap_all_mode=False):
    """验证track帧的人脸数量（支持swap-all模式）"""
    if swap_all_mode:
        # swap-all模式：只需要验证track帧有人脸即可
        swap_face = source_faces['swap_face']
        skip_positions = source_faces['skip_positions']
        
        print(f"[INFO] ========================================")
        print(f"[INFO] Swap-All 模式:")
        print(f"[INFO]   源人脸: 1个（用于所有替换）")
        print(f"[INFO]   跳过位置: {skip_positions if skip_positions else '无'}")
        print(f"[INFO] ========================================")
        print(f"[INFO] 验证 track 帧...")
        
        verify_cap = cv2.VideoCapture(target_path)
        if not verify_cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {target_path}")
            return False
        
        verify_cap.set(cv2.CAP_PROP_POS_FRAMES, track_frame)
        ret, verify_frame = verify_cap.read()
        verify_cap.release()
        
        if not ret or verify_frame is None:
            print(f"[ERROR] 无法读取帧 {track_frame}")
            return False
        
        from core.config import get_face_analyser
        temp_analyser = get_face_analyser()
        detected_faces = temp_analyser.get(verify_frame)
        detected_count = len(detected_faces) if detected_faces else 0
        
        print(f"[INFO] Track 帧 {track_frame} 检测到 {detected_count} 个人脸")
        
        if detected_count == 0:
            print(f"[ERROR] Track 帧未检测到人脸！")
            return False
        
        # 验证跳过位置是否合法
        if skip_positions:
            invalid_positions = [p for p in skip_positions if p >= detected_count]
            if invalid_positions:
                print(f"[ERROR] 跳过位置 {invalid_positions} 超出范围（检测到{detected_count}个人脸，索引0-{detected_count-1}）")
                return False
        
        print(f"[SUCCESS] Swap-All 模式验证通过！")
        print(f"[INFO]   将替换: {detected_count - len(skip_positions)} 个人脸")
        print(f"[INFO]   将跳过: {len(skip_positions)} 个人脸（位置: {skip_positions}）")
        print(f"[INFO] ========================================")
        return True
    
    # 原有的正常模式验证逻辑
    total_sources = len(source_faces)
    skip_count = sum(1 for f in source_faces if f is None)
    normal_count = total_sources - skip_count
    
    print(f"[INFO] ========================================")
    print(f"[INFO] 源人脸配置:")
    print(f"[INFO]   总数: {total_sources}")
    print(f"[INFO]   正常换脸: {normal_count}")
    print(f"[INFO]   跳过(skip): {skip_count}")
    print(f"[INFO] ========================================")
    print(f"[INFO] 重要提示：")
    print(f"[INFO]   - 必须确保 track 帧包含 {total_sources} 个人脸")
    print(f"[INFO]   - 源人脸顺序必须与视频中人脸位置（从左到右）一致")
    print(f"[INFO]   - 如果某个位置不需要换脸，请使用 'skip' 占位")
    print(f"[INFO] ========================================")
    
    print(f"[INFO] 验证 track 帧的人脸数量...")
    print(f"[INFO] 使用帧 {track_frame} 进行验证")
    
    verify_cap = cv2.VideoCapture(target_path)
    if not verify_cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {target_path}")
        return False
    
    verify_cap.set(cv2.CAP_PROP_POS_FRAMES, track_frame)
    ret, verify_frame = verify_cap.read()
    verify_cap.release()
    
    if not ret or verify_frame is None:
        print(f"[ERROR] 无法读取帧 {track_frame}")
        print(f"[INFO] 提示：视频总帧数可能少于 {track_frame}")
        return False
    
    print("[INFO] 正在检测 track 帧的人脸...")
    from core.config import get_face_analyser
    temp_analyser = get_face_analyser()
    detected_faces = temp_analyser.get(verify_frame)
    detected_count = len(detected_faces) if detected_faces else 0
    
    print(f"[INFO] Track 帧 {track_frame} 检测到 {detected_count} 个人脸")
    print(f"[INFO] 提供的源人脸数量: {total_sources}")
    
    if detected_count != total_sources:
        print(f"[ERROR] ========================================")
        print(f"[ERROR] 人脸数量不匹配！")
        print(f"[ERROR]   Track 帧检测到: {detected_count} 个人脸")
        print(f"[ERROR]   提供的源人脸: {total_sources} 个（包括 {skip_count} 个skip）")
        print(f"[ERROR] ========================================")
        print(f"[ERROR] 解决方案：")
        if detected_count < total_sources:
            print(f"[ERROR]   1. 减少源人脸数量到 {detected_count} 个")
            print(f"[ERROR]   2. 或更换 --track-frame 到包含 {total_sources} 个人脸的帧")
        else:
            print(f"[ERROR]   1. 增加源人脸数量到 {detected_count} 个（多余位置用 'skip'）")
            print(f"[ERROR]   2. 或更换 --track-frame 到包含 {total_sources} 个人脸的帧")
        print(f"[ERROR] ========================================")
        return False
    
    print(f"[SUCCESS] 人脸数量验证通过！开始处理...")
    print(f"[INFO] ========================================")
    return True


def main(args):
    # 在导入 config 逻辑生效前设置环境变量
    os.environ["FACE_MODEL"] = args.face_model
    os.environ["is_hybrid_mode()"] = str(args.hybrid)
    print(f"[DEBUG] 已设置环境变量 FACE_MODEL={os.environ['FACE_MODEL']}, is_hybrid_mode()={os.environ['is_hybrid_mode()']}")
    
    # 显示可用模型信息
    model_manager = ModelManager()
    if args.list_models:
        model_manager.list_models()
        return
    
    # 检查模型是否可用
    print(f"[INFO] 检查模型: {args.model}")
    try:
        model_path = model_manager.get_model_path(args.model)
        print(f"[INFO] 模型准备完成: {model_path}")
    except Exception as e:
        print(f"[ERROR] 模型准备失败: {e}")
        return
    
    # 处理swap-all模式
    swap_all_mode = args.swap_all is not None
    skip_positions = []
    if swap_all_mode:
        skip_positions = [int(p) for p in args.swap_all.split(',') if p.strip()]
    
    # 加载源人脸
    source_faces = load_source_faces(args.faces, args.hybrid, swap_all_mode, skip_positions)
    if source_faces is None:
        return
    
    # 图片模式
    if is_img(args.target):
        if swap_all_mode:
            print("[ERROR] swap-all模式不支持图片处理")
            return
        
        process_img(source_faces, args.target, model_name=args.model, 
                   pixel_boost=args.pixel_boost, auto_pixel_boost=args.auto_pixel_boost,
                   debug=args.debug)
        save_to = args.output if args.output else path_safe(
            os.path.splitext(args.target)[0] + "-swapped" + os.path.splitext(args.target)[1]
        )
        try:
            shutil.move(args.target, save_to)
        except Exception:
            os.replace(args.target, save_to)
        print(f"[INFO] 已保存结果到 {save_to}")
        return

    # 视频模式
    output_path = args.output if args.output else "output.mp4"
    
    # 解析时间范围为帧号
    start_frame = args.start_frame
    end_frame = args.end_frame
    
    if start_frame is None and args.start_time:
        from core.utils import parse_time_to_seconds, detect_fps
        fps = detect_fps(args.target)
        seconds = parse_time_to_seconds(args.start_time)
        start_frame = int(seconds * fps)
    
    if end_frame is None and args.end_time:
        from core.utils import parse_time_to_seconds, detect_fps
        fps = detect_fps(args.target)
        seconds = parse_time_to_seconds(args.end_time)
        end_frame = int(seconds * fps)
    
    # 使用带断点续传的直接输出模式（默认推荐）
    if args.checkpoint or args.direct_output or (not args.legacy_direct_output):
        print("[INFO] 使用带断点续传的直接输出模式")
        if args.skip_audio:
            print("[INFO] --skip-audio: 将跳过音频添加阶段")
        
        # 检查 auto_pixel_boost 和 pixel_boost 的冲突
        if args.auto_pixel_boost:
            print("[INFO] 启用自动 Pixel Boost - 将根据每个人脸大小动态调整")
            print("[INFO] 注意：--pixel-boost 参数将被忽略")
        
        # 确定 track 帧号
        if args.track_frame is not None:
            verify_frame_idx = args.track_frame
        elif start_frame is not None:
            verify_frame_idx = start_frame
        else:
            verify_frame_idx = 0
        
        # 验证track帧
        if not verify_track_frame(args.target, verify_frame_idx, source_faces, args.debug, swap_all_mode):
            return
        
        success = process_video_direct_checkpoint(
            source_faces,
            args.target,
            output_path,
            model_name=args.model,
            max_age=args.max_age,
            sim_threshold=args.sim_threshold,
            reset_interval=args.reset_interval,
            pixel_boost=args.pixel_boost,
            segment_frames=args.segment_frames,
            use_multi_gpu=not args.force_single_gpu,
            skip_audio=args.skip_audio,
            auto_pixel_boost=args.auto_pixel_boost,
            debug=args.debug,
            max_workers_per_gpu=args.max_workers_per_gpu,
            start_frame=start_frame,
            end_frame=end_frame,
            track_frame=args.track_frame,
            extract_only=args.extract_only,
            encoder=args.encoder,
            crf=args.crf,
            preset=args.preset,
            swap_all_mode=swap_all_mode  # 新增参数
        )
        
        if success:
            print(f"[INFO] 视频处理完成: {output_path}")
        else:
            print("[ERROR] 视频处理失败或被中断")
        return
    
    # 传统模式
    else:
        if swap_all_mode:
            print("[ERROR] swap-all模式不支持传统模式，请使用checkpoint模式")
            return
        
        print("[INFO] 使用传统帧处理模式")
        if args.skip_audio:
            print("[INFO] --skip-audio: 将跳过音频添加阶段")
        
        from core.utils import detect_fps, extract_frames, add_audio
        import glob
        
        tmp_dir = "frames"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        fps = detect_fps(args.target)
        extract_frames(args.target, tmp_dir)

        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
        print(f"[INFO] 检测到 {len(frame_paths)} 帧，开始处理...")

        # 处理视频帧
        process_video(
            source_faces,
            frame_paths,
            use_multi_gpu=not args.force_single_gpu,
            model_name=args.model,
            max_age=args.max_age,
            sim_threshold=args.sim_threshold,
            reset_interval=args.reset_interval,
            pixel_boost=args.pixel_boost,
            auto_pixel_boost=args.auto_pixel_boost,
            debug=args.debug
        )

        # 合并音频
        add_audio(tmp_dir, args.target, args.keep_frames, output_path, skip_audio=args.skip_audio)
        print(f"[INFO] 生成完成: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roop换脸程序（支持断点续传和自动Pixel Boost，支持部分处理和指定track帧）")
    
    # 人脸检测模型与混合模式
    parser.add_argument("--face-model", type=str, default="buffalo_l",
                       choices=["buffalo_l", "buffalo_sc", "antelopev2"],
                       help="选择人脸检测/识别模型 (默认: buffalo_l)")
    parser.add_argument("--hybrid", action="store_true",
                       help="启用混合检测（antelopev2/buffalo_sc 检测 + buffalo_l embedding）")
    
    # 基本参数
    parser.add_argument("-t", "--target", required=True, help="目标视频或图片")
    parser.add_argument("-f", "--faces", nargs="+", required=True, help="源人脸图片，可以写 skip 表示跳过")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("--keep-frames", action="store_true", help="是否保留处理后的帧")
    parser.add_argument("--keep-fps", action="store_true", help="是否保持原始帧率")
    parser.add_argument("--force-single-gpu", action="store_true", help="强制只用单GPU")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    # Swap-All 模式
    parser.add_argument("--swap-all", type=str, default=None,
                       help="启用swap-all模式：除指定位置外的所有人脸都换成同一个源脸。"
                            "参数为跳过的人脸位置索引（逗号分隔，从0开始），例如: --swap-all 0 表示跳过第1个人脸（从左到右）")
    
    # 音频控制
    parser.add_argument("--skip-audio", action="store_true", help="跳过音频添加阶段，只输出无音频视频")
    
    # 输出模式选择
    parser.add_argument("--checkpoint", action="store_true", 
                       help="使用带断点续传的直接输出模式（推荐，支持中断恢复）")
    parser.add_argument("--direct-output", action="store_true", 
                       help="使用带断点续传的直接输出模式（同--checkpoint）")
    parser.add_argument("--legacy-direct-output", action="store_true",
                       help="使用传统直接输出模式（无断点续传）")
    
    # 断点续传参数
    parser.add_argument("--segment-frames", type=int, default=600,
                       help="每个分段的帧数（默认600帧，约30秒@20fps）")
    
    parser.add_argument("--list-models", action="store_true", help="列出所有可用模型")
    
    # 模型选择参数
    parser.add_argument("--model", default="inswapper_128", 
                       choices=["inswapper_128", "inswapper_128_fp16", "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256"],
                       help="选择换脸模型 (默认: inswapper_128)")
    
    # 跟踪参数
    parser.add_argument("--max-age", type=int, default=70, 
                       help="track最大存活时间（帧数），-1表示永不删除 (默认: 70)")
    parser.add_argument("--sim-threshold", type=float, default=0.16,
                       help="相似度阈值 (默认: 0.16)")
    parser.add_argument("--reset-interval", type=int, default=60,
                       help="跟踪重置间隔（帧数） (默认: 60)")
    
    # Pixel Boost 参数
    parser.add_argument("--pixel-boost", default="128x128",
                       help="Pixel Boost 分辨率，例如 256x256 / 512x512 / 768x768")
    parser.add_argument("--auto-pixel-boost", action="store_true",
                       help="自动根据人脸大小选择最佳 Pixel Boost 参数（推荐）")
    
    # 时间范围参数
    parser.add_argument("--start-time", type=str, default=None,
                       help="开始时间，格式: HH:MM:SS 或 MM:SS 或秒数，例如: 1:30 或 90")
    parser.add_argument("--end-time", type=str, default=None,
                       help="结束时间，格式同上。如果不指定，处理到视频结尾")
    
    # 帧范围参数（优先级高于时间范围）
    parser.add_argument("--start-frame", type=int, default=None,
                       help="开始帧号（从0开始）")
    parser.add_argument("--end-frame", type=int, default=None,
                       help="结束帧号（不包含）")
    
    # track参考帧
    parser.add_argument("--track-frame", type=int, default=None,
                       help="指定用于track的参考帧号。默认使用处理范围的第一帧")
    
    # 是否只输出处理的片段
    parser.add_argument("--extract-only", action="store_true",
                       help="只输出处理的片段，不合并原视频其他部分")

    # 硬件编码选项
    parser.add_argument("--encoder", type=str, default="libx264",
                       choices=["libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_videotoolbox"],
                       help="视频编码器选择 (默认: libx264软件编码。硬件编码选项: h264_nvenc(NVIDIA), hevc_nvenc(NVIDIA HEVC), h264_qsv(Intel), h264_videotoolbox(Mac))")
    parser.add_argument("--crf", type=int, default=23,
                       help="视频质量参数CRF (默认: 23，范围: 0-51，越小质量越好，文件越大)")
    parser.add_argument("--preset", type=str, default="medium",
                       help="编码速度预设 (libx264: ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow; nvenc: slow/medium/fast)")
    
    # Worker数量
    parser.add_argument("--max-workers-per-gpu", type=int, default=4,
                       help="每个GPU最大worker数量（默认4，自动计算时的上限）")
    
    args = parser.parse_args()
    
    # 参数验证
    if args.sim_threshold < 0.0 or args.sim_threshold > 1.0:
        print("[ERROR] sim_threshold必须在0.0-1.0之间")
        exit(1)
    
    if args.max_age < -1:
        print("[ERROR] max_age必须大于等于-1")
        exit(1)
    
    if args.reset_interval < 1:
        print("[ERROR] reset_interval必须大于等于1")
        exit(1)
    
    if args.segment_frames < 10:
        print("[ERROR] segment_frames必须大于等于10")
        exit(1)
    
    # swap-all模式验证
    if args.swap_all is not None:
        if len(args.faces) != 1:
            print("[ERROR] swap-all模式只需要提供1个源人脸")
            exit(1)
    
    # 设置全局环境变量供 config.py 使用
    os.environ["FACE_MODEL"] = args.face_model
    os.environ["is_hybrid_mode()"] = "True" if args.hybrid else "False"
    
    print(f"[DEBUG] 已设置环境变量 FACE_MODEL={os.environ['FACE_MODEL']}, is_hybrid_mode()={os.environ['is_hybrid_mode()']}")
    
    main(args)
