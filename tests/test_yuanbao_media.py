"""
test_yuanbao_media.py - yuanbao_media 单元测试

测试覆盖：
  1. JPEG 尺寸解析：基线 SOF0（公共路径回归保护）
  2. JPEG 尺寸解析：标记前含 0xFF 填充字节（ITU-T T.81 §B.1.1.2）
  3. JPEG 尺寸解析：SOF1（0xC1，基线扩展）
"""

import sys
import os

# 确保 hermes-agent 根目录在 sys.path 中
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import struct

from gateway.platforms.yuanbao_media import parse_image_size


def _make_jpeg(width: int, height: int, sof_marker: int = 0xC0, fill_bytes: int = 0) -> bytes:
    """
    构建一个最小的合法 JPEG 字节串，仅含 SOI + SOF 段 + EOI。

    SOF 段结构（足以让解析器读取宽高）：
      FF <marker> <2 字节段长> <1 字节精度> <2 字节高> <2 字节宽> <分量数据...>
    可在 SOF 标记前插入若干 0xFF 填充字节（合法）。
    """
    soi = b"\xff\xd8"
    # SOF 段载荷：精度(1) + 高(2) + 宽(2) + 1 个分量(3) = 8 字节
    precision = b"\x08"
    h = struct.pack(">H", height)
    w = struct.pack(">H", width)
    component = b"\x01\x11\x00"  # 单分量：id=1, 采样=0x11, 量化表=0
    payload = precision + h + w + component
    seg_len = struct.pack(">H", len(payload) + 2)  # 段长含自身 2 字节
    fill = b"\xff" * fill_bytes
    sof = fill + bytes([0xFF, sof_marker]) + seg_len + payload
    eoi = b"\xff\xd9"
    return soi + sof + eoi


def test_parse_jpeg_size_baseline_sof0():
    """基线 SOF0：公共路径回归保护（main 上即通过）。"""
    data = _make_jpeg(100, 200, sof_marker=0xC0)
    assert parse_image_size(data) == {"width": 100, "height": 200}


def test_parse_jpeg_size_skips_fill_bytes():
    """SOF 标记前的 0xFF 填充字节应被跳过（main 上返回 None，本次修复后通过）。"""
    data = _make_jpeg(100, 200, sof_marker=0xC0, fill_bytes=1)
    assert parse_image_size(data) == {"width": 100, "height": 200}


def test_parse_jpeg_size_sof1():
    """SOF1（0xC1，基线扩展）应被识别（main 上返回 None，本次修复后通过）。"""
    data = _make_jpeg(64, 48, sof_marker=0xC1)
    assert parse_image_size(data) == {"width": 64, "height": 48}
