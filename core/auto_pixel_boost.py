# core/auto_pixel_boost.py
import numpy as np
from typing import Tuple, Optional

class AutoPixelBoostSelector:
    """自动选择 Pixel Boost 参数，根据人脸大小动态调整处理分辨率"""
    
    def __init__(self, model_type: str = 'inswapper'):
        self.model_type = model_type
        
        # 模型配置
        if model_type == 'hyperswap':
            self.base_size = 256
            self.available_options = [256, 512, 768]
        else:  # inswapper
            self.base_size = 128
            self.available_options = [128, 256, 512]
        
        # 人脸尺寸阈值(像素)
        self.size_thresholds = {
            'tiny': 64, 'small': 128, 'medium': 256, 'large': 384, 'xlarge': 512
        }
    
    def calculate_face_size(self, bbox) -> float:
        """计算人脸的有效尺寸(使用几何平均)"""
        if bbox is None or len(bbox) < 4:
            return 0
        
        x1, y1, x2, y2 = bbox[:4]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return np.sqrt(width * height)
    
    def select_pixel_boost(self, bbox, frame_resolution: Optional[Tuple[int, int]] = None) -> str:
        """根据人脸大小自动选择 pixel boost 参数"""
        face_size = self.calculate_face_size(bbox)
        
        # 计算相对大小
        relative_size = 1.0
        if frame_resolution is not None:
            frame_width, frame_height = frame_resolution
            frame_area = frame_width * frame_height
            face_area = face_size * face_size
            relative_size = face_area / frame_area if frame_area > 0 else 0
        
        # 根据人脸尺寸和模型类型选择boost
        if self.model_type == 'inswapper':
            if face_size < 80:
                boost_size = 128
            elif face_size < 160:
                boost_size = 128 if face_size < 120 else 256
            elif face_size < 320:
                boost_size = 256
            elif face_size < 480:
                boost_size = 512 if relative_size > 0.1 else 256
            else:
                boost_size = 512
        else:  # hyperswap
            if face_size < 120:
                boost_size = 256
            elif face_size < 240:
                boost_size = 256
            elif face_size < 400:
                boost_size = 512
            else:
                boost_size = 768 if relative_size > 0.15 else 512
        
        # 确保选择的尺寸在可用选项中
        if boost_size not in self.available_options:
            boost_size = min(self.available_options, key=lambda x: abs(x - boost_size))
        
        return f"{boost_size}x{boost_size}"
    
    def should_use_pixel_boost(self, bbox) -> bool:
        """判断是否需要使用 pixel boost"""
        face_size = self.calculate_face_size(bbox)
        return face_size > self.base_size * 1.5


def get_recommended_pixel_boost(face, model_name: str = 'inswapper_128', 
                                frame_resolution: Optional[Tuple[int, int]] = None) -> str:
    """便捷函数：获取推荐的 pixel boost 参数"""
    model_type = 'hyperswap' if 'hyperswap' in model_name.lower() else 'inswapper'
    selector = AutoPixelBoostSelector(model_type)
    
    if hasattr(face, 'bbox'):
        return selector.select_pixel_boost(face.bbox, frame_resolution)
    return f"{selector.base_size}x{selector.base_size}"
