# core/config.py
import os
import cv2
import numpy as np
import requests
import zipfile
import insightface
import core.globals

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== 动态配置 =====
def get_face_model():
    """从环境变量读取 FACE_MODEL"""
    model = os.environ.get("FACE_MODEL", "buffalo_l")
    if model not in ("buffalo_l", "buffalo_sc", "antelopev2"):
        print(f"[WARN] 未知的 FACE_MODEL: {model}，已自动切换为 buffalo_l")
        return "buffalo_l"
    return model

def is_hybrid_mode():
    """从环境变量读取 is_hybrid_mode"""
    val = os.environ.get("is_hybrid_mode()", "False").lower()
    return val in ("1", "true", "yes")

# ===== antelopev2 修复 =====
def fix_antelopev2_structure(base_dir=None):
    """修复 antelopev2 模型路径和文件名"""
    import shutil
    if base_dir is None:
        base_dir = os.path.expanduser("~/.insightface/models/antelopev2")
    
    # 修复嵌套目录
    nested_dir = os.path.join(base_dir, "antelopev2")
    if os.path.isdir(nested_dir):
        print("[FIX] 检测到多余嵌套目录，正在调整结构...")
        for item in os.listdir(nested_dir):
            s = os.path.join(nested_dir, item)
            d = os.path.join(base_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        shutil.rmtree(nested_dir)
        print("[FIX] 多余目录已删除，目录结构修复完成")
    
    # 修复文件名
    src = os.path.join(base_dir, "scrfd_10g_bnkps.onnx")
    dst = os.path.join(base_dir, "det_10g.onnx")
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print("[FIX] 自动重命名 scrfd_10g_bnkps.onnx -> det_10g.onnx")
    return base_dir

# ===== ModelManager =====
class ModelManager:
    """统一模型下载与管理"""
    MODELS = {
        # 人脸检测模型
        'antelopev2': {
            'folder': 'antelopev2',
            'urls': [
                'https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip',
                'https://sourceforge.net/projects/insightface.mirror/files/v0.7/antelopev2.zip/download',
                'https://huggingface.co/InsighFaceModels/antelopev2/resolve/main/antelopev2.zip'
            ],
            'description': '高精度侧脸检测模型'
        },
        'buffalo_sc': {
            'folder': 'buffalo_sc',
            'urls': [
                'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip',
                'https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_sc.zip/download',
                'https://huggingface.co/InsighFaceModels/buffalo_sc/resolve/main/buffalo_sc.zip'
            ],
            'description': '轻量人脸检测模型'
        },
        'buffalo_l': {
            'folder': 'buffalo_l',
            'urls': [
                'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
                'https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download',
                'https://huggingface.co/InsighFaceModels/buffalo_l/resolve/main/buffalo_l.zip'
            ],
            'description': '旧版人脸检测模型'
        },
        # 换脸模型
        'inswapper_128': {
            'filename': 'inswapper_128.onnx',
            'urls': [
                'https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128.onnx',
                'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'
            ],
            'description': 'INSwapper 128x128 模型'
        },
        'inswapper_128_fp16': {
            'filename': 'inswapper_128_fp16.onnx',
            'urls': [
                'https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx',
                'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128_fp16.onnx'
            ],
            'description': 'INSwapper 128x128 FP16 模型'
        },
        'hyperswap_1a_256': {
            'filename': 'hyperswap_1a_256.onnx',
            'urls': ['https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx'],
            'description': 'HyperSwap 1A 256x256模型'
        },
        'hyperswap_1b_256': {
            'filename': 'hyperswap_1b_256.onnx',
            'urls': ['https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx'],
            'description': 'HyperSwap 1B 256x256模型'
        },
        'hyperswap_1c_256': {
            'filename': 'hyperswap_1c_256.onnx',
            'urls': ['https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx'],
            'description': 'HyperSwap 1C 256x256模型'
        }
    }

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 脚本启动时自动修复 antelopev2
        antelope_dir = os.path.expanduser("~/.insightface/models/antelopev2")
        if os.path.exists(antelope_dir):
            fix_antelopev2_structure(antelope_dir)

    def ensure_model(self, name):
        """检测并下载指定模型"""
        if name not in self.MODELS:
            raise ValueError(f"未知模型: {name}")
        info = self.MODELS[name]

        if 'filename' in info:
            path = os.path.join(self.model_dir, info['filename'])
            if not os.path.exists(path):
                self._try_download(info['urls'], path)
            return path
        else:
            folder_path = os.path.expanduser(f"~/.insightface/models/{info['folder']}")
            if not os.path.exists(folder_path) or not os.listdir(folder_path):
                zip_path = os.path.join(self.model_dir, f"{info['folder']}.zip")
                self._try_download(info['urls'], zip_path)
                self._unzip(zip_path, os.path.dirname(folder_path))
            
            # antelopev2 自动修复
            if name == "antelopev2":
                fix_antelopev2_structure(folder_path)
            return folder_path

    def _try_download(self, urls, save_path):
        """尝试从多个URL下载"""
        for i, url in enumerate(urls):
            try:
                print(f"[INFO] 从 {url} 下载中...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    with open(save_path, "wb") as f:
                        downloaded = 0
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                    print(f"\r[INFO] 下载进度: {downloaded/total*100:.1f}%", end="")
                print(f"\n[INFO] 模型下载完成: {save_path}")
                return
            except Exception as e:
                print(f"\n[WARNING] 从 {url} 下载失败: {e}")
                if i == len(urls)-1:
                    raise RuntimeError(f"[ERROR] 所有地址下载失败: {urls}")

    def _unzip(self, zip_path, target_dir):
        """解压模型"""
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"[INFO] 模型解压完成: {target_dir}")

    def get_model_path(self, name):
        return self.ensure_model(name)

# 单例
model_manager = ModelManager()

# ===== 人脸检测函数 =====
def _ensure_embedding(face):
    """确保face对象有normed_embedding"""
    if face is None:
        return None
    if not hasattr(face, "normed_embedding") or face.normed_embedding is None:
        emb = getattr(face, "embedding", None)
        if emb is not None:
            arr = np.array(emb, dtype=np.float32)
            n = np.linalg.norm(arr)
            face.normed_embedding = arr / n if n > 0 else None
        else:
            face.normed_embedding = None
    return face

def get_face(img, index=0, from_right=False):
    """标准人脸检测"""
    if not hasattr(get_face, "_analyser"):
        ctx_id = 0 if core.globals.use_gpu else -1
        name = get_face_model()
        print(f"[INFO] 初始化人脸检测器: {name} (标准模式)")
        
        os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)
        model_manager.ensure_model(name)
        
        analyser = insightface.app.FaceAnalysis(name=name, providers=core.globals.providers)
        analyser.prepare(ctx_id=ctx_id, det_size=(960, 960))
        get_face._analyser = analyser
        print(f"[INFO] 人脸检测器初始化完成: {name}")
    
    faces = get_face._analyser.get(img)
    
    # 旋转尝试
    if not faces:
        for angle in (-30, 30):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            faces = get_face._analyser.get(rotated)
            if faces:
                print(f"[INFO] 旋转 {angle}° 后检测到人脸")
                break
    
    if not faces:
        return None
    
    faces = sorted(faces, key=lambda x: x.bbox[0])
    if from_right:
        faces.reverse()
    
    return _ensure_embedding(faces[index]) if index < len(faces) else None

def get_face_hybrid(img, index=0, from_right=False):
    """Hybrid 模式人脸检测: antelopev2检测 + buffalo_l生成embedding"""
    from insightface.app.common import Face
    
    # 确保 antelopev2 模型存在并修复目录
    antelope_dir = os.path.expanduser("~/.insightface/models/antelopev2")
    if not os.path.exists(antelope_dir) or not os.path.exists(os.path.join(antelope_dir, "det_10g.onnx")):
        print("[INFO] 下载并修复 antelopev2 模型...")
        model_manager.ensure_model("antelopev2")

    # 初始化检测器和embedding模型
    if not hasattr(get_face_hybrid, "_det_analyser"):
        print("[INFO] 初始化 antelopev2 检测器 (hybrid)")
        get_face_hybrid._det_analyser = insightface.app.FaceAnalysis(
            name="antelopev2",
            providers=core.globals.providers
        )
        get_face_hybrid._det_analyser.prepare(ctx_id=0, det_size=(640, 640))

    if not hasattr(get_face_hybrid, "_emb_analyser"):
        print("[INFO] 初始化 buffalo_l embedding 模型 (hybrid)")
        get_face_hybrid._emb_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=core.globals.providers
        )
        get_face_hybrid._emb_analyser.prepare(ctx_id=0, det_size=(640, 640))

    det_analyser = get_face_hybrid._det_analyser
    emb_analyser = get_face_hybrid._emb_analyser

    # 用 antelopev2 检测人脸
    faces_det = det_analyser.get(img)
    if not faces_det:
        print("[WARN] antelopev2 未检测到人脸")
        return None

    # 选择最合适的脸
    best_face_det = faces_det[0]
    if len(faces_det) > 1:
        h, w = img.shape[:2]
        cx = w / 2
        for f in faces_det:
            fx1, fy1, fx2, fy2 = f.bbox
            f_center = (fx1 + fx2) / 2
            if (from_right and f_center > cx) or (not from_right and f_center < cx):
                best_face_det = f
                break

    # 用 buffalo_l 获取 embedding
    faces_emb = emb_analyser.get(img)
    if not faces_emb:
        print("[WARN] buffalo_l 未检测到人脸")
        return None

    # 匹配位置最接近的 embedding 人脸
    x1, y1, x2, y2 = best_face_det.bbox.astype(int)
    best_face_emb = None
    min_dist = 1e9
    
    for f in faces_emb:
        bx1, by1, bx2, by2 = f.bbox
        dist = np.linalg.norm(
            np.array([(bx1 + bx2) / 2, (by1 + by2) / 2]) -
            np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        )
        if dist < min_dist:
            min_dist = dist
            best_face_emb = f

    if best_face_emb is None:
        print("[WARN] 未能匹配到对应 embedding")
        return None

    # 手动拷贝 Face 对象
    try:
        f = Face()
        f.bbox = best_face_emb.bbox.copy() if hasattr(best_face_emb, "bbox") else None
        f.kps = best_face_emb.kps.copy() if hasattr(best_face_emb, "kps") else None
        f.landmark_3d_68 = getattr(best_face_emb, "landmark_3d_68", None)
        f.embedding = best_face_emb.embedding.copy() if hasattr(best_face_emb, "embedding") else None
        f.det_score = getattr(best_face_emb, "det_score", 1.0)
        f.gender = getattr(best_face_emb, "gender", None)
        f.age = getattr(best_face_emb, "age", None)
        f.pose = getattr(best_face_emb, "pose", None)
        print("[DEBUG] 手动拷贝人脸对象成功。")
        return f
    except Exception as e:
        print(f"[ERROR] 手动拷贝人脸对象失败: {e}")
        return best_face_emb

def get_face_analyser():
    """获取人脸分析器"""
    import torch
    
    name = get_face_model()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        ctx_id = 0
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1
    
    analyser = insightface.app.FaceAnalysis(name=name, providers=providers)
    analyser.prepare(ctx_id=ctx_id, det_size=(960, 960))
    return analyser
