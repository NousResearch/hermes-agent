"""
坐标校正增强工具 - 超级进化6

为 computer_use 提供坐标校正能力，解决截图缩放导致的点击偏移问题。

核心公式：
X_real = X_out · (W_screen / W_img)
Y_real = Y_out · (H_screen / H_img)
"""

from tools.registry import registry
import logging
import subprocess
import json
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def get_screen_size() -> Tuple[int, int]:
    """
    获取屏幕分辨率
    
    Returns:
        (width, height)
    """
    try:
        # macOS
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            displays = data.get('SPDisplaysDataType', [])
            if displays and len(displays) > 0:
                # 获取主显示器
                main_display = displays[0]
                items = main_display.get('spdisplays_ndrvs', [])
                if items and len(items) > 0:
                    resolution = items[0].get('_spdisplays_resolution', '')
                    # 解析 "2560 x 1440" 格式
                    parts = resolution.split('x')
                    if len(parts) == 2:
                        width = int(parts[0].strip())
                        height = int(parts[1].strip())
                        return width, height
        
        # 降级方案：使用默认值
        logger.warning("无法获取屏幕分辨率，使用默认值 2560x1440")
        return 2560, 1440
        
    except Exception as e:
        logger.error(f"获取屏幕分辨率失败: {e}")
        return 2560, 1440


def correct_coordinates(x_out: float, y_out: float, 
                        img_width: int, img_height: int,
                        screen_width: Optional[int] = None,
                        screen_height: Optional[int] = None) -> Tuple[int, int]:
    """
    坐标校正
    
    Args:
        x_out: 输出/图像坐标 X
        y_out: 输出/图像坐标 Y
        img_width: 图像宽度
        img_height: 图像高度
        screen_width: 屏幕宽度（可选，自动检测）
        screen_height: 屏幕高度（可选，自动检测）
        
    Returns:
        (x_real, y_real) 真实屏幕坐标
    """
    # 获取屏幕分辨率
    if screen_width is None or screen_height is None:
        screen_width, screen_height = get_screen_size()
    
    # 应用校正公式
    x_real = x_out * (screen_width / img_width)
    y_real = y_out * (screen_height / img_height)
    
    # 四舍五入到整数
    x_real = round(x_real)
    y_real = round(y_real)
    
    logger.info(f"坐标校正: ({x_out}, {y_out}) @ {img_width}x{img_height} → ({x_real}, {y_real}) @ {screen_width}x{screen_height}")
    
    return x_real, y_real


def coordinate_correction_handler(
    x: float,
    y: float,
    img_width: int,
    img_height: int,
    screen_width: Optional[int] = None,
    screen_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    坐标校正工具处理器
    
    Args:
        x: 输出/图像坐标 X
        y: 输出/图像坐标 Y
        img_width: 图像宽度
        img_height: 图像高度
        screen_width: 屏幕宽度（可选）
        screen_height: 屏幕高度（可选）
        
    Returns:
        校正结果
    """
    try:
        x_real, y_real = correct_coordinates(
            x, y, img_width, img_height,
            screen_width, screen_height
        )
        
        # 获取实际屏幕分辨率
        if screen_width is None or screen_height is None:
            screen_width, screen_height = get_screen_size()
        
        return {
            'success': True,
            'input': {
                'x': x,
                'y': y,
                'img_width': img_width,
                'img_height': img_height
            },
            'output': {
                'x_real': x_real,
                'y_real': y_real,
                'screen_width': screen_width,
                'screen_height': screen_height
            },
            'formula': 'X_real = X_out * (W_screen / W_img); Y_real = Y_out * (H_screen / H_img)',
            'message': f"✅ 坐标校正完成: ({x}, {y}) → ({x_real}, {y_real})"
        }
        
    except Exception as e:
        logger.error(f"坐标校正失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"坐标校正失败: {e}"
        }


# 注册工具
registry.register(
    name="coordinate_correction",
    toolset="computer_use",
    schema={
        "name": "coordinate_correction",
        "description": "坐标校正工具（超级进化6）。解决截图缩放导致的点击偏移问题。输入图像坐标和图像尺寸，输出真实屏幕坐标。",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "输出/图像坐标 X"
                },
                "y": {
                    "type": "number",
                    "description": "输出/图像坐标 Y"
                },
                "img_width": {
                    "type": "integer",
                    "description": "图像宽度（像素）"
                },
                "img_height": {
                    "type": "integer",
                    "description": "图像高度（像素）"
                },
                "screen_width": {
                    "type": "integer",
                    "description": "屏幕宽度（像素，可选，自动检测）"
                },
                "screen_height": {
                    "type": "integer",
                    "description": "屏幕高度（像素，可选，自动检测）"
                }
            },
            "required": ["x", "y", "img_width", "img_height"]
        }
    },
    handler=coordinate_correction_handler
)
