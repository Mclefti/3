"""
Serialization utilities for heightmap/image data.
"""

import numpy as np
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple


def heightmap_to_png_base64(heightmap: np.ndarray) -> str:
    """
    Convert heightmap array to base64-encoded PNG.
    Heightmap values should be in 0-1 range.
    """
    # Scale to 0-255 and convert to uint8
    scaled = (np.clip(heightmap, 0, 1) * 255).astype(np.uint8)
    
    # Create grayscale image
    img = Image.fromarray(scaled, mode='L')
    
    # Encode to PNG and base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def normal_map_to_png_base64(normal_map: np.ndarray) -> str:
    """
    Convert normal map array to base64-encoded PNG.
    Normal map should have shape (H, W, 3) with values in 0-1 range.
    """
    # Scale to 0-255
    scaled = (np.clip(normal_map, 0, 1) * 255).astype(np.uint8)
    
    # Create RGB image
    img = Image.fromarray(scaled, mode='RGB')
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def splat_map_to_png_base64(splat_map: np.ndarray) -> str:
    """
    Convert splat map array to a false-colour RGB PNG for browser preview.
    Splat map should have shape (H, W, 5) with values in 0-1 range.
    Channels: [0]=water, [1]=sand, [2]=grass, [3]=rock, [4]=snow.

    Each channel weight is blended against its representative colour so the
    output is always fully opaque and matches the 3D shader's palette.
    """
    water = splat_map[..., 0]
    sand  = splat_map[..., 1]
    grass = splat_map[..., 2]
    rock  = splat_map[..., 3]
    snow  = splat_map[..., 4]

    # Representative colours — mirrored from the 3D terrain shader
    water_col = np.array([0.10, 0.30, 0.50])   # deep blue
    sand_col  = np.array([0.76, 0.70, 0.50])   # sandy yellow
    grass_col = np.array([0.20, 0.50, 0.20])   # mid green
    rock_col  = np.array([0.40, 0.35, 0.30])   # warm brown-grey
    snow_col  = np.array([0.95, 0.95, 1.00])   # near-white

    # Weighted blend → shape (H, W, 3)
    rgb = (
        water[..., np.newaxis] * water_col +
        sand[...,  np.newaxis] * sand_col  +
        grass[..., np.newaxis] * grass_col +
        rock[...,  np.newaxis] * rock_col  +
        snow[...,  np.newaxis] * snow_col
    )

    scaled = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(scaled, mode='RGB')

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_heightmap(b64_string: str) -> np.ndarray:
    """
    Decode base64 PNG back to heightmap array.
    Returns values in 0-1 range.
    """
    buffer = BytesIO(base64.b64decode(b64_string))
    img = Image.open(buffer).convert('L')
    
    return np.array(img).astype(np.float64) / 255.0
