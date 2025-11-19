# core/globals.py
import onnxruntime
import torch
import subprocess

use_gpu = True
gpu_count = 0
providers = ['CPUExecutionProvider']

def _check_gpu_runtime():
    """检查GPU是否可用"""
    if not torch.cuda.is_available():
        return False
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

# 配置GPU
available_providers = onnxruntime.get_available_providers()
if use_gpu and 'CUDAExecutionProvider' in available_providers and _check_gpu_runtime():
    gpu_count = torch.cuda.device_count()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"[INFO] 检测到 {gpu_count} 个GPU设备")
else:
    use_gpu = False
    print("[INFO] 使用CPUExecutionProvider")

max_gpu_workers = min(4, gpu_count) if gpu_count > 0 else 1
use_multi_gpu = gpu_count > 1 and max_gpu_workers > 1

print(f"[INFO] use_gpu={use_gpu}, gpu_count={gpu_count}, use_multi_gpu={use_multi_gpu}, max_gpu_workers={max_gpu_workers}")
