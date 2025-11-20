# Roop 换脸程序 - 完整使用文档

## 📋 目录

- [简介](#简介)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [高级功能](#高级功能)
- [参数详解](#参数详解)
- [常见问题](#常见问题)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 简介

这是一个基于深度学习的高性能人脸替换工具，支持图片和视频处理。采用先进的人脸追踪算法和自适应相似度匹配，能够在复杂场景下实现稳定的人脸替换效果。

### 核心优势

- ✅ **智能人脸追踪** - 自适应相似度阈值，支持表情变化和遮挡恢复
- ✅ **断点续传** - 支持中断后继续处理，节省时间
- ✅ **多GPU加速** - 自动平衡GPU显存，智能分配worker数量
- ✅ **灵活的人脸控制** - 支持跳过特定位置的人脸（skip机制）
- ✅ **自动超分辨率** - Auto Pixel Boost根据人脸大小自动调整处理质量
- ✅ **部分视频处理** - 支持指定时间/帧范围处理
- ✅ **混合检测模式** - antelopev2高精度检测 + buffalo_l特征提取

---

## 功能特性

### 1. 人脸检测模型
- **buffalo_l** - 标准模型，速度快
- **buffalo_sc** - 轻量模型，显存占用少
- **antelopev2** - 高精度模型，侧脸检测效果更好

### 2. 换脸模型
- **inswapper_128** - 标准模型（128x128基础分辨率）
- **inswapper_128_fp16** - FP16精度版本，显存占用更少
- **hyperswap_1a/1b/1c_256** - 高质量模型（256x256基础分辨率）

### 3. Pixel Boost（超分辨率）
- 手动模式：128x128, 256x256, 512x512, 768x768
- 自动模式：根据人脸大小智能选择最佳分辨率

### 4. 编码器支持
- **libx264** - 软件编码（通用，兼容性最好）
- **h264_nvenc** - NVIDIA GPU硬件编码
- **hevc_nvenc** - NVIDIA HEVC编码
- **h264_qsv** - Intel QuickSync硬件编码
- **h264_videotoolbox** - macOS硬件编码

---

## 环境要求

### 硬件要求
- **CPU**: 任意现代处理器
- **内存**: 至少8GB RAM（推荐16GB+）
- **GPU**: 
  - NVIDIA GPU（推荐，6GB+ 显存）
  - 或仅使用CPU（处理速度较慢）

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11
- **CUDA**: 11.x 或 12.x（如果使用GPU）
- **FFmpeg**: 4.x 或更新版本

---

## 安装步骤

### 1. 安装Python依赖

```bash
# 克隆或下载项目
cd roop-project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装基础依赖
pip install opencv-python numpy requests

# 安装深度学习框架
# GPU版本（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# 或CPU版本
pip install torch torchvision
pip install onnxruntime

# 安装InsightFace
pip install insightface

# 其他依赖
pip install psutil
```

### 2. 安装FFmpeg

**Windows:**
```bash
# 使用Chocolatey
choco install ffmpeg

# 或从官网下载: https://ffmpeg.org/download.html
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. 验证安装

```bash
# 检查Python环境
python --version
python -c "import torch; print(torch.cuda.is_available())"

# 检查FFmpeg
ffmpeg -version

# 列出可用模型
python run.py --list-models
```

---

## 快速开始

### 图片换脸

```bash
# 基础用法
python run.py -t target.jpg -f source.jpg -o output.jpg

# 使用自动Pixel Boost
python run.py -t target.jpg -f source.jpg -o output.jpg --auto-pixel-boost

# 使用高质量模型
python run.py -t target.jpg -f source.jpg -o output.jpg --model hyperswap_1a_256
```

### 视频换脸

```bash
# 基础用法（自动使用断点续传）
python run.py -t video.mp4 -f face1.jpg -o output.mp4

# 多人换脸
python run.py -t video.mp4 -f face1.jpg face2.jpg face3.jpg -o output.mp4

# 跳过第二个人脸（保持原样）
python run.py -t video.mp4 -f face1.jpg skip face3.jpg -o output.mp4

# 使用自动Pixel Boost
python run.py -t video.mp4 -f face1.jpg -o output.mp4 --auto-pixel-boost
```

### 处理视频片段

```bash
# 按时间范围处理
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --start-time 1:30 --end-time 3:45

# 按帧号处理
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --start-frame 100 --end-frame 500

# 只输出处理的片段（不合并原视频）
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --start-time 1:30 --end-time 2:00 --extract-only
```

---

## 详细使用说明

### 1. 多人换脸顺序

**重要**: 源人脸的顺序必须与视频中人脸的位置（从左到右）对应。

```bash
# 例如：视频中有3个人，从左到右分别是A、B、C
# 如果要替换A和C，保持B不变：
python run.py -t video.mp4 -f face_A.jpg skip face_C.jpg -o output.mp4
```

### 2. Track帧的概念

Track帧是用于建立人脸追踪的参考帧。程序会在这一帧检测所有人脸，并为每个人脸建立追踪。

```bash
# 自动使用第一帧作为track帧
python run.py -t video.mp4 -f face1.jpg face2.jpg -o output.mp4

# 指定track帧（当第一帧人脸不清晰时）
python run.py -t video.mp4 -f face1.jpg face2.jpg -o output.mp4 --track-frame 50

# 处理片段时，track帧默认使用片段第一帧
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --start-frame 100 --end-frame 500 --track-frame 150
```

### 3. 断点续传机制

程序会自动保存处理进度，支持中断后继续：

```bash
# 首次运行
python run.py -t video.mp4 -f face.jpg -o output.mp4

# 如果中断（Ctrl+C 或错误），直接重新运行相同命令即可继续
python run.py -t video.mp4 -f face.jpg -o output.mp4

# 临时文件位置：output_segments/
# 检查点文件：output_segments/checkpoint.json
```

**手动清理临时文件：**
```bash
# 如果需要重新开始
rm -rf output_segments/  # Linux/Mac
rmdir /s output_segments  # Windows
```

### 4. 混合检测模式

混合模式使用antelopev2进行高精度检测，buffalo_l提取特征：

```bash
# 启用混合模式
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --face-model antelopev2 --hybrid

# 适用场景：
# - 侧脸较多的视频
# - 人脸被部分遮挡
# - 远距离或小人脸
```

---

## 高级功能

### 1. GPU显存管理

程序会自动检测显存并分配worker数量：

```bash
# 限制每个GPU的worker数量（避免显存不足）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-workers-per-gpu 2

# 强制使用单GPU（即使有多个GPU）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --force-single-gpu
```

**显存占用估算：**
- 模型基础: ~1.5GB
- 每个worker: ~1.7GB
- GPU0额外开销: ~1.9GB

### 2. 编码器选择

```bash
# 使用NVIDIA硬件编码（最快）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --encoder h264_nvenc

# 调整质量和速度
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --encoder h264_nvenc --preset fast --crf 20

# CRF值说明：
# 0-17: 视觉无损（文件很大）
# 18-23: 高质量（推荐，23为默认）
# 24-28: 中等质量
# 29+: 低质量
```

### 3. 跟踪参数调优

```bash
# 放宽相似度阈值（适合表情变化大的场景）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --sim-threshold 0.12

# 增加track存活时间（适合人脸被遮挡的场景）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-age 120

# 禁用track自动重置（适合稳定场景）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --reset-interval -1
```

### 4. 分段大小调整

```bash
# 减小分段大小（减少显存占用，但增加文件I/O）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --segment-frames 300

# 增大分段大小（减少文件I/O，但需要更多显存）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --segment-frames 1200
```

---

## 参数详解

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --target` | 目标图片或视频 | 必填 |
| `-f, --faces` | 源人脸图片列表（可包含skip） | 必填 |
| `-o, --output` | 输出文件路径 | 自动生成 |
| `--debug` | 启用调试模式 | False |

### 模型参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--face-model` | 人脸检测模型 | buffalo_l, buffalo_sc, antelopev2 | buffalo_l |
| `--hybrid` | 混合检测模式 | - | False |
| `--model` | 换脸模型 | inswapper_128, hyperswap_1a_256等 | inswapper_128 |

### Pixel Boost参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--pixel-boost` | 超分辨率 | 128x128, 256x256, 512x512, 768x768 | 128x128 |
| `--auto-pixel-boost` | 自动选择Pixel Boost | - | False |

### 跟踪参数

| 参数 | 说明 | 范围 | 默认值 |
|------|------|------|--------|
| `--sim-threshold` | 相似度阈值 | 0.0-1.0 | 0.16 |
| `--max-age` | track最大存活帧数（-1=永不删除） | -1或正整数 | 70 |
| `--reset-interval` | track重置间隔帧数 | 正整数 | 60 |

### 范围参数

| 参数 | 说明 | 格式 |
|------|------|------|
| `--start-time` | 开始时间 | HH:MM:SS 或 MM:SS 或秒数 |
| `--end-time` | 结束时间 | 同上 |
| `--start-frame` | 开始帧号 | 整数（从0开始） |
| `--end-frame` | 结束帧号 | 整数（不包含） |
| `--track-frame` | Track参考帧 | 整数 |
| `--extract-only` | 只输出片段 | - |

### 编码参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--encoder` | 视频编码器 | libx264, h264_nvenc等 | libx264 |
| `--crf` | 质量参数 | 0-51 | 23 |
| `--preset` | 编码速度 | ultrafast~veryslow | medium |

### 性能参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-workers-per-gpu` | 每GPU最大worker数 | 4 |
| `--force-single-gpu` | 强制单GPU | False |
| `--segment-frames` | 分段大小（帧数） | 600 |

### 其他参数

| 参数 | 说明 |
|------|------|
| `--skip-audio` | 跳过音频处理 |
| `--keep-frames` | 保留临时帧文件 |
| `--list-models` | 列出所有可用模型 |

---

## 常见问题

### Q1: 如何确定源人脸的顺序？

**A:** 打开目标视频的某一帧，观察人脸从左到右的位置，源人脸列表必须按照这个顺序提供。

示例：
```
视频中从左到右：Alice, Bob, Charlie
命令应该是：-f alice.jpg bob.jpg charlie.jpg

如果只想替换Alice和Charlie：
-f alice.jpg skip charlie.jpg
```

### Q2: 程序中断后如何继续？

**A:** 直接重新运行相同的命令即可，程序会自动从上次中断的地方继续。

### Q3: 显存不足（OOM）怎么办？

**A:** 尝试以下方法：
1. 减少worker数量：`--max-workers-per-gpu 2`
2. 使用FP16模型：`--model inswapper_128_fp16`
3. 减小Pixel Boost：`--pixel-boost 128x128`
4. 减小分段大小：`--segment-frames 300`

### Q4: 换脸效果不理想怎么办？

**A:** 
1. 使用高质量模型：`--model hyperswap_1a_256`
2. 启用自动Pixel Boost：`--auto-pixel-boost`
3. 使用混合检测模式：`--face-model antelopev2 --hybrid`
4. 调整相似度阈值：`--sim-threshold 0.12`（更低=更宽松）

### Q5: 如何处理多个人但只换其中几个？

**A:** 使用`skip`占位：
```bash
# 视频中有5个人，只换第1、3、5个
python run.py -t video.mp4 -f face1.jpg skip face3.jpg skip face5.jpg -o output.mp4
```

### Q6: Track帧选择有什么建议？

**A:** 
- 选择所有目标人脸都清晰可见的帧
- 避免选择有遮挡或模糊的帧
- 使用`--track-frame`参数指定合适的帧号

### Q7: 音频处理失败怎么办？

**A:** 
1. 使用`--skip-audio`跳过音频处理
2. 检查FFmpeg是否正确安装
3. 原视频可能没有音频轨道

---

## 性能优化

### 1. 硬件加速

**使用NVIDIA GPU硬件编码：**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --encoder h264_nvenc --preset fast
```

**使用Intel QSV：**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --encoder h264_qsv
```

### 2. Worker数量优化

**自动分配（推荐）：**
```bash
# 程序会根据显存自动计算最佳worker数量
python run.py -t video.mp4 -f face.jpg -o output.mp4
```

**手动调整：**
```bash
# 显存充足时增加worker
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-workers-per-gpu 6

# 显存不足时减少worker
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-workers-per-gpu 2
```

### 3. 处理速度 vs 质量权衡

**最快速度（质量较低）：**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model inswapper_128_fp16 \
  --pixel-boost 128x128 \
  --encoder h264_nvenc --preset fast --crf 28
```

**平衡模式（推荐）：**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model inswapper_128 \
  --auto-pixel-boost \
  --encoder h264_nvenc --preset medium --crf 23
```

**最高质量（速度较慢）：**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model hyperswap_1a_256 \
  --pixel-boost 512x512 \
  --encoder libx264 --preset slow --crf 18
```

### 4. 大视频处理建议

对于超长视频（>30分钟）：

```bash
# 1. 分段处理（例如每10分钟一段）
python run.py -t video.mp4 -f face.jpg -o part1.mp4 \
  --start-time 0:00 --end-time 10:00 --extract-only

python run.py -t video.mp4 -f face.jpg -o part2.mp4 \
  --start-time 10:00 --end-time 20:00 --extract-only

# 2. 使用FFmpeg合并
ffmpeg -i part1.mp4 -i part2.mp4 -i part3.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[outv][outa]" \
  -map "[outv]" -map "[outa]" final.mp4
```

---

## 故障排除

### 问题：ImportError: No module named 'cv2'

**解决：**
```bash
pip install opencv-python
```

### 问题：CUDA out of memory

**解决：**
```bash
# 方法1: 减少worker数量
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-workers-per-gpu 1

# 方法2: 使用CPU模式
python run.py -t video.mp4 -f face.jpg -o output.mp4 --force-single-gpu
# 同时在core/globals.py中设置 use_gpu = False

# 方法3: 使用FP16模型
python run.py -t video.mp4 -f face.jpg -o output.mp4 --model inswapper_128_fp16
```

### 问题：人脸数量不匹配

**错误信息：**
```
[ERROR] 人脸数量不匹配！
  Track 帧检测到: 3 个人脸
  提供的源人脸: 2 个
```

**解决：**
```bash
# 方法1: 添加skip占位
python run.py -t video.mp4 -f face1.jpg face2.jpg skip -o output.mp4

# 方法2: 更换track帧
python run.py -t video.mp4 -f face1.jpg face2.jpg -o output.mp4 --track-frame 100
```

### 问题：FFmpeg not found

**解决：**

Windows:
```bash
choco install ffmpeg
# 或手动添加到PATH环境变量
```

Linux:
```bash
sudo apt install ffmpeg
```

macOS:
```bash
brew install ffmpeg
```

### 问题：换脸效果闪烁

**解决：**
```bash
# 增加相似度阈值（更严格匹配）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --sim-threshold 0.20

# 增加track存活时间
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-age 120

# 减少重置频率
python run.py -t video.mp4 -f face.jpg -o output.mp4 --reset-interval 120
```

### 问题：某些人脸没有被替换

**解决：**
```bash
# 方法1: 降低相似度阈值
python run.py -t video.mp4 -f face.jpg -o output.mp4 --sim-threshold 0.12

# 方法2: 使用混合检测模式
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --face-model antelopev2 --hybrid

# 方法3: 禁用track过期（不推荐用于长视频）
python run.py -t video.mp4 -f face.jpg -o output.mp4 --max-age -1
```

### 问题：处理速度很慢

**诊断：**
```bash
# 检查是否使用GPU
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
python -c "import torch; print('GPU数量:', torch.cuda.device_count())"
```

**优化：**
```bash
# 1. 确保使用GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# 2. 使用硬件编码
python run.py -t video.mp4 -f face.jpg -o output.mp4 --encoder h264_nvenc

# 3. 降低质量设置
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model inswapper_128_fp16 --pixel-boost 128x128 --crf 28
```

---

## 命令行示例集合

### 基础示例

```bash
# 1. 单人换脸（图片）
python run.py -t photo.jpg -f source.jpg -o output.jpg

# 2. 单人换脸（视频）
python run.py -t video.mp4 -f source.jpg -o output.mp4

# 3. 多人换脸
python run.py -t video.mp4 -f person1.jpg person2.jpg person3.jpg -o output.mp4

# 4. 跳过特定位置
python run.py -t video.mp4 -f person1.jpg skip person3.jpg -o output.mp4
```

### 高质量处理

```bash
# 5. 使用高质量模型 + 自动超分
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --model hyperswap_1a_256 --auto-pixel-boost

# 6. 手动指定高分辨率
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --model hyperswap_1a_256 --pixel-boost 768x768

# 7. 使用混合检测（侧脸场景）
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --face-model antelopev2 --hybrid --auto-pixel-boost
```

### 片段处理

```bash
# 8. 处理1分30秒到3分45秒
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --start-time 1:30 --end-time 3:45

# 9. 处理100-500帧
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --start-frame 100 --end-frame 500

# 10. 只提取处理片段
python run.py -t video.mp4 -f source.jpg -o clip.mp4 \
  --start-time 1:00 --end-time 2:00 --extract-only
```

### 性能优化

```bash
# 11. 使用NVIDIA硬件编码
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --encoder h264_nvenc --preset fast

# 12. 限制显存使用
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --max-workers-per-gpu 2 --segment-frames 300

# 13. 快速处理模式
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --model inswapper_128_fp16 --pixel-boost 128x128 \
  --encoder h264_nvenc --preset fast --crf 28
```

### 特殊场景

```bash
# 14. 表情变化大的场景
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --sim-threshold 0.12 --max-age 120

# 15. 人脸被遮挡的场景
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --max-age 150 --sim-threshold 0.14

# 16. 稳定场景（减少闪烁）
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --sim-threshold 0.20 --reset-interval 120

# 17. 指定特定帧作为track参考
python run.py -t video.mp4 -f source.jpg -o output.mp4 \
  --track-frame 50
```

### 调试和测试

```bash
# 18. 启用调试模式
python run.py -t video.mp4 -f source.jpg -o output.mp4 --debug

# 19. 无音频输出
python run.py -t video.mp4 -f source.jpg -o output.mp4 --skip-audio

# 20. 保留临时帧文件
DEBUG_KEEP_TEMP=1 python run.py -t video.mp4 -f source.jpg -o output.mp4
```

---

## 项目结构

```
roop-project/
├── run.py                      # 主程序入口
├── core/
│   ├── __init__.py
│   ├── globals.py              # 全局配置（GPU设置）
│   ├── config.py               # 模型管理和人脸检测
│   ├── processor.py            # 核心处理逻辑
│   ├── checkpoint_manager.py   # 断点续传管理
│   ├── processing_range.py     # 视频范围处理
│   ├── auto_pixel_boost.py     # 自动超分辨率
│   └── utils.py                # 工具函数
├── models/                     # 模型文件目录（自动下载）
├── README.md                   # 本文档
└── requirements.txt            # Python依赖列表
```

---

## 技术原理

### 1. 人脸检测与追踪

程序使用InsightFace进行人脸检测和特征提取：

- **检测阶段**: 使用SCRFD检测人脸位置和关键点
- **特征提取**: 使用ArcFace提取512维人脸特征向量
- **追踪算法**: 基于余弦相似度的自适应追踪
  - 初始阈值: 0.16（可调整）
  - 自适应调整: 根据表情变化动态放宽阈值
  - 丢失恢复: 支持短暂遮挡后的追踪恢复

### 2. 换脸过程

1. **人脸对齐**: 使用5点关键点进行仿射变换
2. **特征融合**: 混合源人脸和目标人脸的embedding
3. **神经网络推理**: 使用ONNX模型生成换脸结果
4. **超分辨率**: 可选的Pixel Boost提升细节
5. **自然融合**: 使用渐变mask融合到原图

### 3. Pixel Boost原理

Pixel Boost通过分块处理实现超分辨率：

```
输入: 512x512高分辨率人脸区域
↓
分割为: 4x4 = 16个 128x128的块
↓
每块独立处理: 128x128 → 换脸 → 128x128
↓
重组: 16个块 → 512x512高质量结果
```

**自动模式**根据人脸大小选择最佳分辨率：
- 小人脸(<80px): 128x128
- 中等人脸(80-320px): 256x256
- 大人脸(>320px): 512x512

### 4. 断点续传机制

- **分段处理**: 视频分为固定大小的segment（默认600帧）
- **进度保存**: 每完成一个segment自动保存到checkpoint.json
- **中断恢复**: 自动检测已完成的segment，跳过重复处理
- **验证机制**: 启动时验证已有segment的完整性

---

## 性能基准

### 硬件配置参考

| 配置 | GPU | 处理速度 | 显存占用 |
|------|-----|---------|---------|
| 入门 | GTX 1660 6GB | ~8 FPS | 4-5GB |
| 推荐 | RTX 3060 12GB | ~15 FPS | 6-8GB |
| 高性能 | RTX 4090 24GB | ~35 FPS | 10-12GB |
| CPU模式 | 无GPU | ~1-2 FPS | RAM 8GB+ |

*基于1080p视频，inswapper_128模型，256x256 Pixel Boost*

### 多GPU性能

| GPU数量 | 理论加速比 | 实际加速比 |
|---------|-----------|-----------|
| 1x GPU | 1.0x | 1.0x |
| 2x GPU | 2.0x | 1.8x |
| 4x GPU | 4.0x | 3.2x |

*实际性能受视频I/O和CPU处理影响*

---

## 模型对比

### 人脸检测模型

| 模型 | 精度 | 速度 | 侧脸检测 | 推荐场景 |
|------|-----|------|---------|---------|
| buffalo_l | ★★★☆ | ★★★★ | ★★☆☆ | 通用场景 |
| buffalo_sc | ★★☆☆ | ★★★★★ | ★★☆☆ | 显存受限 |
| antelopev2 | ★★★★★ | ★★★☆ | ★★★★★ | 侧脸/遮挡 |

### 换脸模型

| 模型 | 质量 | 速度 | 显存 | 推荐用途 |
|------|-----|------|------|---------|
| inswapper_128 | ★★★☆ | ★★★★ | 低 | 快速处理 |
| inswapper_128_fp16 | ★★★☆ | ★★★★★ | 极低 | 显存受限 |
| hyperswap_1a_256 | ★★★★ | ★★★☆ | 中 | 高质量 |
| hyperswap_1b_256 | ★★★★☆ | ★★☆☆ | 中 | 最高质量 |
| hyperswap_1c_256 | ★★★★★ | ★★☆☆ | 中高 | 专业级 |

---

## 最佳实践

### 1. 源人脸图片准备

**推荐规格：**
- 分辨率: 512x512 或更高
- 格式: JPG/PNG
- 人脸角度: 正面，±15度内
- 光照: 均匀，避免强阴影
- 表情: 自然中性表情
- 清晰度: 高清，无模糊

**不推荐：**
- ❌ 低分辨率图片(<256x256)
- ❌ 过度侧脸(>30度)
- ❌ 强烈表情(大笑、怒目等)
- ❌ 戴口罩或大面积遮挡
- ❌ 艺术照、美颜过度

### 2. 目标视频选择

**适合处理：**
- ✅ 正脸或小角度侧脸
- ✅ 光照稳定
- ✅ 人脸清晰可见
- ✅ 分辨率720p或以上

**处理困难：**
- ⚠️ 快速运动场景
- ⚠️ 频繁切换镜头
- ⚠️ 极端光照变化
- ⚠️ 长时间大角度侧脸

### 3. 参数选择建议

**场景：正面对话视频**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model inswapper_128 \
  --auto-pixel-boost \
  --sim-threshold 0.16
```

**场景：侧脸较多**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --face-model antelopev2 --hybrid \
  --model hyperswap_1a_256 \
  --auto-pixel-boost \
  --sim-threshold 0.14
```

**场景：快速运动/表情丰富**
```bash
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --model hyperswap_1a_256 \
  --auto-pixel-boost \
  --sim-threshold 0.12 \
  --max-age 100 \
  --reset-interval 40
```

**场景：多人群像**
```bash
python run.py -t video.mp4 -f p1.jpg p2.jpg p3.jpg -o output.mp4 \
  --face-model antelopev2 --hybrid \
  --model inswapper_128 \
  --auto-pixel-boost \
  --sim-threshold 0.16 \
  --track-frame 50
```

### 4. 质量检查流程

1. **小范围测试**
   ```bash
   # 先处理10秒测试效果
   python run.py -t video.mp4 -f face.jpg -o test.mp4 \
     --start-time 0:10 --end-time 0:20 --extract-only
   ```

2. **检查关键帧**
   - 人脸是否准确替换
   - 边缘是否自然
   - 是否有闪烁
   - 肤色是否匹配

3. **调整参数**
   - 如有闪烁 → 提高sim-threshold
   - 如有漏检 → 降低sim-threshold或使用混合模式
   - 如质量不佳 → 提高pixel-boost或使用hyperswap

4. **全片处理**
   ```bash
   # 使用最优参数处理完整视频
   python run.py -t video.mp4 -f face.jpg -o final.mp4 \
     [最优参数]
   ```

---

## 常见错误代码

| 错误代码 | 含义 | 解决方案 |
|---------|------|---------|
| `FileNotFoundError: target.mp4` | 找不到目标文件 | 检查文件路径 |
| `CUDA out of memory` | 显存不足 | 减少worker或使用FP16 |
| `Face not detected` | 未检测到人脸 | 检查图片质量或使用混合模式 |
| `Face count mismatch` | 人脸数量不匹配 | 调整源人脸列表或track-frame |
| `FFmpeg not found` | FFmpeg未安装 | 安装FFmpeg |
| `Checkpoint corrupted` | 检查点损坏 | 删除临时文件重新开始 |

---

## 更新日志

### v2.0.0 (2024-01)
- ✨ 新增自动Pixel Boost功能
- ✨ 新增混合检测模式（antelopev2 + buffalo_l）
- ✨ 新增部分视频处理支持
- ✨ 新增skip机制（跳过特定人脸）
- ⚡ 优化GPU显存管理，自动分配worker数量
- ⚡ 改进断点续传机制
- 🐛 修复多人换脸时的track错位问题
- 🐛 修复长视频处理时的显存泄漏

### v1.5.0 (2023-12)
- ✨ 添加hyperswap模型支持
- ✨ 添加硬件编码器支持
- ⚡ 优化人脸追踪算法
- 🐛 修复音频同步问题

### v1.0.0 (2023-10)
- 🎉 首次发布
- ✨ 基础换脸功能
- ✨ 断点续传支持
- ✨ 多GPU加速

---

## 许可证

本项目仅供学习和研究使用。

**重要声明：**
- 禁止用于制作虚假信息、诈骗或任何非法用途
- 使用本工具处理他人肖像需获得授权
- 用户需对使用本工具产生的内容负责
- 开发者不对滥用行为承担任何责任

---

## 贡献指南

欢迎提交Issue和Pull Request！

### 报告Bug
请提供以下信息：
1. 操作系统和Python版本
2. GPU型号和驱动版本
3. 完整的错误信息
4. 复现步骤
5. 命令行参数

### 功能建议
请描述：
1. 功能需求
2. 使用场景
3. 预期效果

---

## 致谢

本项目基于以下开源项目：
- [InsightFace](https://github.com/deepinsight/insightface) - 人脸检测和识别
- [ONNX Runtime](https://onnxruntime.ai/) - 模型推理
- [FFmpeg](https://ffmpeg.org/) - 视频处理

---

## 联系方式

- 问题反馈: [GitHub Issues]
- 技术交流: [论坛/Discord链接]

---

## FAQ扩展

### Q8: 如何批量处理多个视频？

**A:** 使用脚本：
```bash
#!/bin/bash
# batch_process.sh

for video in *.mp4; do
    echo "处理 $video ..."
    python run.py -t "$video" -f face.jpg -o "output_$video"
done
```

### Q9: 如何提取最佳track帧？

**A:** 使用FFmpeg提取关键帧：
```bash
# 提取视频的前10个关键帧
ffmpeg -i video.mp4 -vf "select=eq(pict_type\,I)" \
  -vsync vfr -frames:v 10 frame_%03d.jpg

# 人工检查哪一帧所有人脸都清晰
# 假设frame_005.jpg最合适，它是第150帧
# 使用: --track-frame 150
```

### Q10: 如何评估处理质量？

**A:** 检查要点：
1. **边缘融合**: 换脸区域与背景的过渡是否自然
2. **肤色匹配**: 脸部肤色与脖子/耳朵是否一致
3. **光照一致性**: 脸部光照方向是否与场景匹配
4. **表情自然度**: 面部表情是否协调
5. **时间连贯性**: 连续帧之间是否流畅，无闪烁

### Q11: 如何处理超长视频（1小时+）？

**A:** 建议分段处理：
```bash
# 1. 先测试一小段确定参数
python run.py -t video.mp4 -f face.jpg -o test.mp4 \
  --start-time 10:00 --end-time 10:30 --extract-only

# 2. 分段处理（每20分钟一段）
python run.py -t video.mp4 -f face.jpg -o part1.mp4 \
  --start-time 0:00 --end-time 20:00 --extract-only

python run.py -t video.mp4 -f face.jpg -o part2.mp4 \
  --start-time 20:00 --end-time 40:00 --extract-only

python run.py -t video.mp4 -f face.jpg -o part3.mp4 \
  --start-time 40:00 --end-time 60:00 --extract-only

# 3. 合并所有片段
ffmpeg -f concat -safe 0 -i <(for f in part*.mp4; do echo "file '$PWD/$f'"; done) \
  -c copy final.mp4
```

### Q12: GPU利用率不高怎么办？

**A:** 诊断和优化：
```bash
# 1. 监控GPU使用
watch -n 1 nvidia-smi

# 2. 如果GPU利用率<70%，尝试增加worker
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --max-workers-per-gpu 6

# 3. 如果瓶颈在CPU，减少segment大小加快写入
python run.py -t video.mp4 -f face.jpg -o output.mp4 \
  --segment-frames 300

# 4. 检查是否I/O瓶颈
# - 使用SSD而非HDD
# - 确保有足够的磁盘空间
```

---

## 附录

### A. 配置文件示例

创建 `config.json` 用于常用配置：
```json
{
  "model": "hyperswap_1a_256",
  "auto_pixel_boost": true,
  "face_model": "antelopev2",
  "hybrid": true,
  "sim_threshold": 0.14,
  "max_age": 80,
  "encoder": "h264_nvenc",
  "preset": "medium",
  "crf": 23
}
```

使用配置文件（需要修改代码支持）：
```python
# 在run.py中添加
import json
with open('config.json') as f:
    config = json.load(f)
    # 应用配置...
```

### B. 环境变量

可用的环境变量：

```bash
# 保留临时文件用于调试
export DEBUG_KEEP_TEMP=1

# 指定模型下载目录
export MODEL_DIR=/path/to/models

# CUDA设备选择
export CUDA_VISIBLE_DEVICES=0,1

# OMP线程数（CPU模式）
export OMP_NUM_THREADS=8
```

### C. 术语表

| 术语 | 说明 |
|------|------|
| **Embedding** | 人脸特征向量，512维 |
| **Track** | 人脸追踪轨迹 |
| **Segment** | 视频分段，用于断点续传 |
| **Pixel Boost** | 超分辨率处理技术 |
| **CRF** | 恒定质量因子，控制视频质量 |
| **Worker** | 并行处理的工作线程 |
| **Checkpoint** | 处理进度检查点 |
| **Skip** | 跳过特定人脸的替换 |

---

**文档版本**: v2.0.0  
**最后更新**: 2024年1月  
**维护者**: [Your Name/Team]

---

*本文档持续更新中，如有疑问或建议，欢迎提Issue！*