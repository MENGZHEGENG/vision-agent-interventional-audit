from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

COLORS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange')

COLOR_RGB = {'red': (220, 55, 55), 'blue': (55, 105, 220), 'green': (45, 165, 90), 'yellow': (230, 185, 45), 'purple': (145, 75, 190), 'orange': (230, 120, 35)}

BACKGROUND = (248, 248, 248)

WIDTH = 256

HEIGHT = 192

def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 4294967295)

def encode_png(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Encode an RGB image without Pillow or another external dependency."""
    if not pixels or not pixels[0]:
        raise ValueError('PNG image must be non-empty')
    height = len(pixels)
    width = len(pixels[0])
    if any((len(row) != width for row in pixels)):
        raise ValueError('PNG rows must have equal widths')
    scanlines = bytearray()
    for row in pixels:
        scanlines.append(0)
        for red, green, blue in row:
            scanlines.extend((red, green, blue))
    header = b'\x89PNG\r\n\x1a\n'
    header += _png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    header += _png_chunk(b'IDAT', zlib.compress(bytes(scanlines), level=9))
    return header + _png_chunk(b'IEND', b'')

def _inside_triangle(px: int, py: int, points: tuple[tuple[int, int], ...]) -> bool:
    (ax, ay), (bx, by), (cx, cy) = points
    first = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    second = (px - bx) * (cy - by) - (py - by) * (cx - bx)
    third = (px - cx) * (ay - cy) - (py - cy) * (ax - cx)
    return first >= 0 and second >= 0 and (third >= 0) or (first <= 0 and second <= 0 and (third <= 0))

def _paint_object(pixels: list[list[tuple[int, int, int]]], obj: dict[str, object], color_name: str | None=None) -> None:
    color = COLOR_RGB[color_name or str(obj['color'])]
    x = int(obj['x'])
    y = int(obj['y'])
    width = int(obj['width'])
    height = int(obj['height'])
    shape = str(obj['shape'])
    x_end = min(WIDTH - 1, x + width)
    y_end = min(HEIGHT - 1, y + height)
    for py in range(max(0, y), y_end + 1):
        for px in range(max(0, x), x_end + 1):
            inside = False
            if shape == 'circle':
                dx = (px - (x + width / 2)) / max(1, width / 2)
                dy = (py - (y + height / 2)) / max(1, height / 2)
                inside = dx * dx + dy * dy <= 1.0
            elif shape == 'triangle':
                inside = _inside_triangle(px, py, ((x + width // 2, y), (x, y_end), (x_end, y_end)))
            else:
                inside = True
            if inside:
                pixels[py][px] = color

def render_scene(scene: dict[str, object], hidden_ids: set[str] | None=None) -> bytes:
    hidden_ids = hidden_ids or set()
    pixels = [[BACKGROUND for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for obj in scene['objects']:
        if str(obj['id']) not in hidden_ids:
            _paint_object(pixels, obj)
    return encode_png(pixels)
