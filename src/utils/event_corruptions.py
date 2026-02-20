# -*- coding: utf-8 -*-
"""
Event-based image corruption functions.

Adapted from imagecorruptions (https://github.com/bethgelab/imagecorruptions)
for use with time-binned event camera frames (2-channel: positive/negative polarity).

API mirrors imagecorruptions for easy comparison:
    from src.utils.event_corruptions import corrupt_event, get_corruption_names
    
    corrupted = corrupt_event(event_frame, corruption_name='gaussian_noise', severity=3)

Input format: numpy array of shape (H, W) for grayscale or (H, W, 2) for polarity channels
Output format: same shape as input, values clipped to [0, 255]
"""

import numpy as np
import math
from PIL import Image
from io import BytesIO
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
import skimage as sk


# /////////////// Corruption Registry ///////////////

CORRUPTION_FUNCTIONS = {}

def register_corruption(name: str, category: str):
    """Decorator to register a corruption function."""
    def decorator(func):
        CORRUPTION_FUNCTIONS[name] = {
            'func': func,
            'category': category
        }
        return func
    return decorator


def get_corruption_names(subset: str = 'common') -> list:
    """
    Get list of available corruption names.
    
    Args:
        subset: One of 'common', 'noise', 'blur', 'weather', 'digital', 'all'
        
    Returns:
        List of corruption names
    """
    categories = {
        'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
        'blur': ['gaussian_blur', 'defocus_blur', 'motion_blur', 'zoom_blur', 'glass_blur'],
        'weather': ['fog', 'frost', 'snow', 'spatter'],
        'digital': ['contrast', 'brightness', 'pixelate', 'jpeg_compression', 'elastic_transform'],
    }
    
    if subset == 'all':
        return list(CORRUPTION_FUNCTIONS.keys())
    elif subset == 'common':
        # Common corruptions (15 from original paper)
        return (categories['noise'][:3] + categories['blur'][:4] + 
                ['snow', 'frost', 'fog', 'brightness', 'contrast', 
                 'elastic_transform', 'pixelate', 'jpeg_compression'])
    elif subset in categories:
        return [c for c in categories[subset] if c in CORRUPTION_FUNCTIONS]
    else:
        raise ValueError(f"Unknown subset: {subset}. Use 'common', 'noise', 'blur', 'weather', 'digital', or 'all'")


def corrupt_event(x, severity: int = 1, corruption_name: str = None, corruption_number: int = None):
    """
    Apply corruption to an event frame.
    
    Args:
        x: Event frame as numpy array. Shape (H, W) for grayscale or (H, W, 2) for polarity.
        severity: Corruption severity level (1-5)
        corruption_name: Name of corruption to apply
        corruption_number: Index of corruption (alternative to name)
        
    Returns:
        Corrupted event frame as numpy array (same shape as input)
    """
    if corruption_name is None and corruption_number is None:
        raise ValueError("Must specify either corruption_name or corruption_number")
    
    if corruption_number is not None:
        names = get_corruption_names('all')
        if corruption_number >= len(names):
            raise ValueError(f"corruption_number {corruption_number} out of range (max {len(names)-1})")
        corruption_name = names[corruption_number]
    
    if corruption_name not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}. Available: {list(CORRUPTION_FUNCTIONS.keys())}")
    
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")
    
    x = np.array(x)
    return CORRUPTION_FUNCTIONS[corruption_name]['func'](x, severity)


# /////////////// Corruption Helpers ///////////////

def _ensure_2channel(x):
    """Ensure input has 2 channels (H, W, 2). Returns (array, was_single_channel)."""
    if x.ndim == 2:
        # Grayscale: duplicate to simulate pos/neg polarity
        return np.stack([x, 255 - x], axis=-1), True
    elif x.ndim == 3 and x.shape[2] == 2:
        return x, False
    elif x.ndim == 3 and x.shape[2] == 1:
        x = x.squeeze(-1)
        return np.stack([x, 255 - x], axis=-1), True
    else:
        raise ValueError(f"Expected shape (H,W) or (H,W,2), got {x.shape}")


def _restore_channels(x, was_single_channel):
    """Restore to original channel format."""
    if was_single_channel:
        return x[:, :, 0]  # Return just the positive channel
    return x


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """Create disk kernel for defocus blur."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                          stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def clipped_zoom(img, zoom_factor):
    """Zoom image and clip to original size."""
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)
    return img


def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1


def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)


def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k / Z


def shift(image, dx, dy):
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:, dx:] = shifted[:, dx-1:dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx:dx+1]
    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:, :] = shifted[dy-1:dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy:dy+1, :]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    """Apply motion blur to single-channel image."""
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred


# /////////////// TIER 1: Direct Application (Trivial) ///////////////

@register_corruption('gaussian_noise', 'noise')
def gaussian_noise_event(x, severity=1):
    """
    Add Gaussian noise to event frame.
    Directly analogous to sensor noise in DVS cameras.
    """
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Apply noise independently to each polarity channel
    noisy = x + np.random.normal(size=x.shape, scale=c)
    
    result = np.clip(noisy, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('shot_noise', 'noise')
def shot_noise_event(x, severity=1):
    """
    Apply Poisson (shot) noise to event frame.
    Models photon shot noise, highly relevant to DVS sensors.
    """
    c = [60, 25, 12, 5, 3][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Apply Poisson noise per channel
    noisy = np.random.poisson(x * c) / float(c)
    
    result = np.clip(noisy, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('impulse_noise', 'noise')
def impulse_noise_event(x, severity=1):
    """
    Apply salt-and-pepper (impulse) noise to event frame.
    Simulates hot/dead pixels in DVS sensors.
    """
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    
    # Apply s&p noise to each channel
    result = np.zeros_like(x, dtype=np.float64)
    for ch in range(2):
        result[:, :, ch] = sk.util.random_noise(
            np.array(x[:, :, ch]) / 255., 
            mode='s&p', 
            amount=c
        )
    
    result = np.clip(result, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('speckle_noise', 'noise')
def speckle_noise_event(x, severity=1):
    """
    Apply multiplicative Gaussian (speckle) noise to event frame.
    """
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Multiplicative noise: x + x * noise
    noisy = x + x * np.random.normal(size=x.shape, scale=c)
    
    result = np.clip(noisy, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('gaussian_blur', 'blur')
def gaussian_blur_event(x, severity=1):
    """
    Apply Gaussian blur to event frame.
    Simulates optical blur / defocus.
    """
    c = [1, 2, 3, 4, 6][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Apply blur to each channel
    blurred = np.zeros_like(x)
    for ch in range(2):
        blurred[:, :, ch] = gaussian(x[:, :, ch], sigma=c)
    
    result = np.clip(blurred, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('pixelate', 'digital')
def pixelate_event(x, severity=1):
    """
    Pixelate event frame by downsampling then upsampling.
    """
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x_shape = x.shape
    
    # Convert to PIL for resizing
    # Process each channel separately
    result = np.zeros_like(x)
    for ch in range(2):
        pil_img = Image.fromarray(x[:, :, ch].astype(np.uint8))
        small = pil_img.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)
        large = small.resize((x_shape[1], x_shape[0]), Image.NEAREST)
        result[:, :, ch] = np.array(large)
    
    return _restore_channels(result, was_single)


@register_corruption('jpeg_compression', 'digital')
def jpeg_compression_event(x, severity=1):
    """
    Apply JPEG compression artifacts to event frame.
    Only relevant if storing events as compressed images.
    """
    c = [25, 18, 15, 10, 7][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    
    # Process each channel
    result = np.zeros_like(x)
    for ch in range(2):
        pil_img = Image.fromarray(x[:, :, ch].astype(np.uint8))
        output = BytesIO()
        pil_img.save(output, 'JPEG', quality=c)
        result[:, :, ch] = np.array(Image.open(output))
    
    return _restore_channels(result, was_single)


# /////////////// TIER 2: Minor Adaptation (Easy) ///////////////

@register_corruption('contrast', 'digital')
def contrast_event(x, severity=1):
    """
    Reduce contrast of event frame.
    Affects event "strength" representation - scales magnitudes around mean.
    """
    c = [0.4, .3, .2, .1, .05][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Scale around per-channel mean
    for ch in range(2):
        mean = np.mean(x[:, :, ch])
        x[:, :, ch] = (x[:, :, ch] - mean) * c + mean
    
    result = np.clip(x, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('brightness', 'digital')
def brightness_event(x, severity=1):
    """
    Adjust brightness of event frame.
    For events, this shifts the activation threshold - use sparingly.
    Reduced severity compared to RGB (50% of original values).
    """
    # Reduced severity for events (they're change-based, not absolute intensity)
    c = [.05, .1, .15, .2, .25][severity - 1]  # Half of RGB values
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    # Add brightness offset
    x = x + c
    
    result = np.clip(x, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('defocus_blur', 'blur')
def defocus_blur_event(x, severity=1):
    """
    Apply defocus blur using disk kernel.
    Same kernel applied to both polarity channels.
    """
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x) / 255.
    
    kernel = disk(radius=c[0], alias_blur=c[1])
    
    # Apply same kernel to both channels
    result = np.zeros_like(x)
    for ch in range(2):
        result[:, :, ch] = cv2.filter2D(x[:, :, ch], -1, kernel)
    
    result = np.clip(result, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('elastic_transform', 'digital')
def elastic_transform_event(x, severity=1):
    """
    Apply elastic deformation to event frame.
    Identical warp applied to both channels to maintain consistency.
    """
    x, was_single = _ensure_2channel(x)
    image = np.array(x, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    sigma = np.array(shape_size) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    # Generate displacement field once
    dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
    dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)

    # Apply same displacement to both channels
    x_coords, y_coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    result = np.zeros_like(image)
    for ch in range(2):
        indices = (
            np.reshape(y_coords + dy, (-1, 1)),
            np.reshape(x_coords + dx, (-1, 1))
        )
        result[:, :, ch] = map_coordinates(
            image[:, :, ch], indices, order=1, mode='reflect'
        ).reshape(shape[:2])
    
    result = np.clip(result, 0, 1) * 255
    return _restore_channels(result, was_single)


# /////////////// TIER 3: Moderate Adaptation (Medium) ///////////////

@register_corruption('motion_blur', 'blur')
def motion_blur_event(x, severity=1):
    """
    Apply motion blur to event frame.
    Reduced severity since event cameras inherently handle motion better.
    Events are sparse at motion boundaries - blur is applied to cumulative frame.
    """
    # Reduced severity for events (70% of RGB parameters)
    c = [(7, 2), (10, 4), (10, 6), (10, 8), (14, 10)][severity - 1]
    
    x, was_single = _ensure_2channel(x)
    x = np.array(x, dtype=np.float32)
    
    angle = np.random.uniform(-45, 45)
    
    # Apply motion blur to each channel
    result = np.zeros_like(x)
    for ch in range(2):
        result[:, :, ch] = _motion_blur(x[:, :, ch], radius=c[0], sigma=c[1], angle=angle)
    
    result = np.clip(result, 0, 255)
    return _restore_channels(result, was_single)


@register_corruption('zoom_blur', 'blur')
def zoom_blur_event(x, severity=1):
    """
    Apply radial zoom blur to event frame.
    Reduced effect since events are less affected by this in practice.
    """
    # Reduced zoom factors for events
    c = [np.arange(1, 1.08, 0.01),
         np.arange(1, 1.12, 0.01),
         np.arange(1, 1.16, 0.02),
         np.arange(1, 1.20, 0.02),
         np.arange(1, 1.24, 0.03)][severity - 1]

    x, was_single = _ensure_2channel(x)
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)

    for zoom_factor in c:
        zoom_layer = clipped_zoom(x, zoom_factor)
        zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]
        try:
            out += zoom_layer
        except ValueError:
            out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

    x = (x + out) / (len(c) + 1)
    result = np.clip(x, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('fog', 'weather')
def fog_event(x, severity=1):
    """
    Add fog effect to event frame.
    In event cameras, fog causes uniform low-contrast events.
    Adds uniform noise floor + plasma fractal pattern to both channels.
    """
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    x, was_single = _ensure_2channel(x)
    shape = x.shape
    max_side = np.max(shape[:2])
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    # Generate fog pattern and apply to both channels
    fog_pattern = plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]]
    
    for ch in range(2):
        x[:, :, ch] += c[0] * fog_pattern

    result = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('glass_blur', 'blur')
def glass_blur_event(x, severity=1):
    """
    Apply glass blur (local pixel shuffling + blur) to event frame.
    Simulates scattering through frosted glass.
    """
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x, was_single = _ensure_2channel(x)
    
    # Initial blur
    blurred = np.zeros_like(x, dtype=np.float64)
    for ch in range(2):
        blurred[:, :, ch] = gaussian(np.array(x[:, :, ch]) / 255., sigma=c[0])
    x = np.uint8(blurred * 255)
    
    x_shape = x.shape

    # Locally shuffle pixels (same shuffling for both channels to maintain consistency)
    for i in range(c[2]):
        for h in range(x_shape[0] - c[1], c[1], -1):
            for w in range(x_shape[1] - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # Swap both channels together
                for ch in range(2):
                    x[h, w, ch], x[h_prime, w_prime, ch] = x[h_prime, w_prime, ch], x[h, w, ch]

    # Final blur
    result = np.zeros_like(x, dtype=np.float64)
    for ch in range(2):
        result[:, :, ch] = gaussian(x[:, :, ch] / 255., sigma=c[0])
    
    result = np.clip(result, 0, 1) * 255
    return _restore_channels(result, was_single)


# /////////////// TIER 4: Significant Rethinking (Hard) ///////////////

@register_corruption('frost', 'weather')
def frost_event(x, severity=1):
    """
    Add frost effect to event frame.
    For events: spatially-correlated stuck-pixel patterns / baseline noise.
    Instead of overlaying frost images, generates procedural frost texture.
    """
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]

    x, was_single = _ensure_2channel(x)
    x_shape = x.shape
    
    # Generate procedural frost pattern using multiple plasma fractals
    map_size = next_power_of_2(max(x_shape[0], x_shape[1]))
    
    frost_pattern = np.zeros((x_shape[0], x_shape[1]), dtype=np.float64)
    # Combine multiple scales
    for scale in [2, 3, 4]:
        fractal = plasma_fractal(mapsize=map_size, wibbledecay=scale)
        frost_pattern += fractal[:x_shape[0], :x_shape[1]]
    frost_pattern = frost_pattern / 3.0
    
    # Apply frost with thresholding to create "crystalline" structure
    frost_pattern = np.clip(frost_pattern, 0.3, 0.7)  # Create mid-range values
    frost_pattern = (frost_pattern - 0.3) / 0.4  # Normalize to [0, 1]
    frost_pattern = frost_pattern * 255
    
    # Blend frost with both channels
    result = np.zeros_like(x, dtype=np.float64)
    for ch in range(2):
        result[:, :, ch] = c[0] * np.array(x[:, :, ch]) + c[1] * frost_pattern
    
    result = np.clip(result, 0, 255)
    return _restore_channels(result, was_single)


@register_corruption('snow', 'weather')
def snow_event(x, severity=1):
    """
    Add snow effect to event frame.
    For events: random transient spikes + directional motion streaks.
    Snow flakes create brief events as they pass through the field of view.
    """
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x, was_single = _ensure_2channel(x)
    x = np.array(x, dtype=np.float32) / 255.

    # Generate snow layer as random events
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    
    # Zoom to create varying snowflake sizes
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

    # Apply motion blur to simulate falling snow (diagonal direction)
    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], 
                               angle=np.random.uniform(-135, -45))
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[:x.shape[0], :x.shape[1]]

    # Blend with original (snow triggers events in both polarities)
    result = np.zeros_like(x)
    for ch in range(2):
        # Snow increases activity
        blended = c[6] * x[:, :, ch] + (1 - c[6]) * np.maximum(x[:, :, ch], x[:, :, ch] * 1.5 + 0.5)
        result[:, :, ch] = blended + snow_layer + np.rot90(snow_layer, k=2)[:x.shape[0], :x.shape[1]]
    
    result = np.clip(result, 0, 1) * 255
    return _restore_channels(result, was_single)


@register_corruption('spatter', 'weather')
def spatter_event(x, severity=1):
    """
    Apply spatter (water droplets / mud) effect to event frame.
    For events: localized gain modulation + local lens distortion.
    Water droplets cause local refraction and event suppression/enhancement.
    """
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

    x, was_single = _ensure_2channel(x)
    x = np.array(x, dtype=np.float32) / 255.

    # Generate liquid/spatter layer
    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    if c[5] == 0:
        # Water spatter - create refractive regions
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        
        m = (liquid_layer.astype(np.float32) * dist.astype(np.float32))
        m /= (np.max(m) + 1e-8)
        m *= c[4]
        
        # Apply to both channels as gain modulation
        result = np.zeros_like(x)
        for ch in range(2):
            result[:, :, ch] = np.clip(x[:, :, ch] + m * 0.5, 0, 1)
    else:
        # Mud spatter - suppress events in affected regions
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        
        result = np.zeros_like(x)
        for ch in range(2):
            result[:, :, ch] = x[:, :, ch] * (1 - m)
            # Add mud color (reduces event activity)
            result[:, :, ch] = np.clip(result[:, :, ch] + 0.2 * m, 0, 1)
    
    result = result * 255
    return _restore_channels(result, was_single)


# /////////////// Convenience Functions ///////////////

def corrupt_sequence(sequence, corruption_name: str, severity: int = 1, 
                     consistent: bool = True, seed: int = None):
    """
    Apply corruption to a sequence of event frames.
    
    Args:
        sequence: List of event frames or array of shape (T, H, W) or (T, H, W, 2)
        corruption_name: Name of corruption to apply
        severity: Corruption severity level (1-5)
        consistent: If True, use same random parameters for all frames
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted sequence in same format as input
    """
    if seed is not None:
        np.random.seed(seed)
    
    if isinstance(sequence, np.ndarray):
        is_array = True
        frames = [sequence[i] for i in range(sequence.shape[0])]
    else:
        is_array = False
        frames = sequence
    
    if consistent:
        # Set seed before first frame, then increment for temporal variation
        base_seed = seed if seed is not None else np.random.randint(0, 2**31)
        corrupted = []
        for i, frame in enumerate(frames):
            np.random.seed(base_seed + i)
            corrupted.append(corrupt_event(frame, severity=severity, 
                                          corruption_name=corruption_name))
    else:
        corrupted = [corrupt_event(f, severity=severity, 
                                   corruption_name=corruption_name) for f in frames]
    
    if is_array:
        return np.stack(corrupted, axis=0)
    return corrupted
