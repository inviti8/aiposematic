"""Aiposematic: Scramble & Recover using image key + operation string , By: Fibo Metavinci"""

__version__ = "0.1"
import os
import numpy as np
import cv2
from PIL import Image
import tempfile
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
import math
import random
import qrcode
from enum import Enum, auto
import base64
from cryptography.fernet import Fernet
import secrets

class SCRAMBLE_MODE(Enum):
    """Available modes for key generation in the scrambling process."""
    NONE = auto()       # No key generation
    BUTTERFLY = auto()  # Use butterfly pattern for key generation
    QR = auto()         # Use QR code for key generation

# ------------------------------------------------------------------
# Aiposematic: Scramble & Recover using image key + operation string
# ------------------------------------------------------------------

def _encrypt_b64_as_string(img_path: str, key: str = None) -> tuple:
    """
    Load an image, convert it to base64, and encrypt it.
    
    Args:
        img_path: Path to the image file
        key: Optional encryption key (32-byte URL-safe base64-encoded bytes).
             If None, a new key will be generated.
             
    Returns:
        tuple: (encrypted_base64_string, key_used)
    """
    # Read the image file as binary data
    with open(img_path, 'rb') as img_file:
        img_data = img_file.read()
    
    # Convert binary data to base64 string
    b64_string = base64.b64encode(img_data).decode('utf-8')
    
    # Generate a key if none provided
    if key is None:
        key = Fernet.generate_key()
    elif not isinstance(key, bytes):
        key = key.encode('utf-8')
    
    # Encrypt the base64 string
    f = Fernet(key)
    encrypted = f.encrypt(b64_string.encode('utf-8'))
    
    return encrypted.decode('utf-8'), key.decode('utf-8')

def _decrypt_b64_image(encrypted_data: str, key: str) -> bytes:
    """
    Decrypt an encrypted base64 image string back to binary data.
    
    Args:
        encrypted_data: The encrypted base64 string
        key: The encryption key used
        
    Returns:
        bytes: The original image binary data
    """
    if not isinstance(key, bytes):
        key = key.encode('utf-8')
    
    f = Fernet(key)
    try:
        # First decode the base64 string
        encrypted_bytes = base64.b64decode(encrypted_data)
        # Then decrypt the bytes
        decrypted = f.decrypt(encrypted_bytes)
        return decrypted
    except Exception as e:
        print(f"Decryption error: {str(e)}")
        print(f"Key length: {len(key) if key else 0} bytes")
        print(f"Encrypted data length: {len(encrypted_data) if encrypted_data else 0} chars")
        raise


def _b64_to_tmp_img(b64_str: str, suffix: str = '.png') -> str:
    """
    Convert a base64 encoded image string to a temporary image file.
    
    Args:
        b64_str: Base64 encoded image string
        suffix: File extension for the temporary file (default: '.png')
        
    Returns:
        str: Path to the temporary image file
    """
    
    # Decode the base64 string to binary data
    try:
        img_data = base64.b64decode(b64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {str(e)}")
    
    # Create a temporary file with the specified suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(img_data)
        temp_path = temp_file.name
    
    return temp_path

def _steganography_encode(original_img_path, key_img_path, output_path=None):
    """
    Hide a key image inside an original image using LSB steganography.
    Uses all 8 bits of each color channel to preserve maximum data.
    
    Args:
        original_img_path: Path to the original image that will hide the key
        key_img_path: Path to the key image to hide
        output_path: Optional output path for the encoded image
        
    Returns:
        str: Path to the encoded image
    """
    # Read the images in RGB mode to ensure consistent channel ordering
    original_img = cv2.imread(original_img_path, cv2.IMREAD_UNCHANGED)
    key_img = cv2.imread(key_img_path, cv2.IMREAD_UNCHANGED)
    
    if original_img is None:
        raise ValueError("Could not read original image")
    if key_img is None:
        raise ValueError("Could not read key image")
    
    # Convert to RGB if they're in BGR
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    if len(key_img.shape) == 3 and key_img.shape[2] == 3:
        key_img = cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB)
    
    # Ensure key image matches original dimensions
    key_img = cv2.resize(key_img, (original_img.shape[1], original_img.shape[0]))
    
    # Create output image with same dimensions and type as original
    encoded_img = np.zeros_like(original_img)
    
    # For each color channel, copy the original pixel values
    # This ensures we don't lose any data from the original image
    for c in range(3):  # RGB channels
        encoded_img[:,:,c] = original_img[:,:,c]
    
    # Save the result with maximum quality
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    # Save as PNG with maximum compression (no loss)
    cv2.imwrite(output_path, cv2.cvtColor(encoded_img, cv2.COLOR_RGB2BGR), 
               [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Now append the key image data to the PNG file
    with open(output_path, 'ab') as f:  # 'ab' mode appends binary data
        with open(key_img_path, 'rb') as key_file:
            f.write(b'APOS' + key_file.read())
    
    return output_path

def _steganography_decode(encoded_img_path, output_path=None):
    """
    Extract the hidden key image from an encoded image.
    
    Args:
        encoded_img_path: Path to the image with hidden data
        output_path: Optional output path for the extracted key image
        
    Returns:
        str: Path to the extracted key image
    """
    # First, read the original image
    img = cv2.imread(encoded_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read encoded image")
    
    # Create output path if not provided
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    # Read the appended data
    with open(encoded_img_path, 'rb') as f:
        # Read the entire file
        data = f.read()
        
        # Find the start of the appended key image data
        # We look for our marker 'APOS' that we added during encoding
        marker = b'APOS'
        marker_pos = data.rfind(marker)
        
        if marker_pos == -1:
            raise ValueError("No hidden data found in the image")
        
        # Extract the key image data
        key_data = data[marker_pos + len(marker):]
        
        # Write the key image data to a temporary file
        with open(output_path, 'wb') as key_file:
            key_file.write(key_data)
    
    # Verify the extracted key image is valid
    key_img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if key_img is None:
        raise ValueError("Invalid key image data in the encoded file")
    
    return output_path

# Precompute a permutation of 0-255 for better diffusion
_permutation = np.random.permutation(256).astype(np.uint8)
_inv_permutation = np.zeros(256, dtype=np.uint8)
for i, val in enumerate(_permutation):
    _inv_permutation[val] = i

def _random_rgb():
    return (random.random(), random.random(), random.random())

def _random_rgb_int():
    """Return a random RGB color as a tuple of integers (0-255)."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def _generate_butterfly(points=1000, size=100, offset_x=0, offset_y=0):
    """Generate butterfly curve points using parametric equations.
    
    Args:
        points: Number of points to generate
        size: Scaling factor for the butterfly size
        offset_x, offset_y: Position offset for the butterfly
        
    Returns:
        x, y: Arrays of x and y coordinates
    """
    l = random.randint(24, 64)  # Random value between 24 and 128
    _x = random.uniform(1.5, 3.0)  # Random float between 1.5 and 3.0
    _y = random.uniform(2.5, 3.5)  # Random float between 1.5 and 3.0
    t = np.linspace(0, l*math.pi, points)
    x = size * (np.sin(t) * (np.exp(np.cos(t)) - _x*np.cos(3*t) - np.sin(t/2)**8)) + offset_x
    y = -size * (np.cos(t) * (np.exp(np.cos(t)) - _y*np.cos(2*t) - np.sin(t/18)**3)) + offset_y
    return x, y

def _generate_butterflies(n=5, size_range=(0.1, 1.0), canvas_size=1000, min_padding=0.05):
    """Generate multiple butterflies with random positions and sizes.
    
    Args:
        n: Number of butterflies to generate
        size_range: Tuple of (min_size, max_size) as a fraction of canvas size
        canvas_size: Size of the canvas
        
    Returns:
        List of (x, y, size) tuples for each butterfly
    """
    butterflies = []
    min_size = int(canvas_size * size_range[0] * 0.05)  # Convert to butterfly size units
    max_size = int(canvas_size * size_range[1] * 0.05)
    
    for _ in range(n):
        size = random.uniform(min_size, max_size)
        # Allow more overlap by reducing padding
        padding = size * min_padding
        x = random.uniform(-canvas_size/2 + padding, canvas_size/2 - padding)
        y = random.uniform(-canvas_size/2 + padding, canvas_size/2 - padding)
        butterflies.append((x, y, size))
    
    return butterflies

def _generate_background(ax, size=100, n_boxes=20, colors=None, dark_theme=False):
    """Generate a background with noise and QR code-like pattern.
    
    Args:
        ax: Matplotlib axis to draw on
        size: Size of the background
        n_boxes: Number of boxes in each dimension for QR pattern
        colors: List of colors to use for the pattern
        dark_theme: Whether to use dark theme colors
    """
    # 1. Add simple colored noise pattern (darker version)
    if dark_theme:
        colors = [
            _random_rgb(),
            _random_rgb(),
            _random_rgb(),
            _random_rgb(),
            _random_rgb()
        ]
        bg_color = _random_rgb()
    else:
        colors = [
            _random_rgb(),
            _random_rgb(),
            _random_rgb(),
            _random_rgb(),
            _random_rgb()
        ]
        bg_color = _random_rgb()
    
    # Create simple colored noise with darker tones
    noise_size = 100  # Lower resolution for more visible noise
    noise = np.random.rand(noise_size, noise_size, 3) * 0.7  # Darker base
    
    # Display the noise pattern
    ax.imshow(noise, 
              extent=[-size/2, size/2, -size/2, size/2], 
              aspect='auto',
              zorder=0,
              interpolation='nearest')  # Keep pixels sharp
    
    # 2. Add simple grid pattern
    box_size = size / n_boxes
    patches = []
    
    # Add fewer, more visible boxes
    for i in range(n_boxes):
        for j in range(n_boxes):
            if random.random() > 0.9:  # 10% chance of a box
                x = -size/2 + i * box_size
                y = -size/2 + j * box_size
                size_variation = 0.5 + random.random() * 0.5  # 50-100% size
                alpha = 0.3  # Fixed opacity
                
                box = Rectangle((x, y), 
                              box_size * size_variation, 
                              box_size * size_variation,
                              color=random.choice(colors),
                              alpha=alpha,
                              zorder=1)
                patches.append(box)
    
    # Add all patches at once for better performance
    if patches:  # Only add if there are patches to avoid warning
        ax.add_collection(PatchCollection(patches, match_original=True))

def _plot_butterflies(butterflies, save_path=None, show=True, width_px=2000, height_px=None, canvas_size=1000):
    """Plot multiple butterflies.
    
    Args:
        butterflies: List of (x, y, size) tuples for each butterfly
        save_path: Path to save the image
        show: Whether to display the plot
        width_px: Output image width in pixels (default: 2000)
        height_px: Output image height in pixels. If None, uses same as width (square)
        canvas_size: Internal canvas size for butterfly coordinates
    """
    if height_px is None:
        height_px = width_px  # Default to square if height not specified
        
    dpi = 100  # Dots per inch
    fig_width = width_px / dpi  # Convert pixels to inches
    fig_height = height_px / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure axes
    
    # Add background with noise and QR pattern for dark theme
    _generate_background(ax, size=canvas_size,
                      colors=[_random_rgb(), _random_rgb(), _random_rgb()],
                      dark_theme=True)
    
    # Plot each butterfly
    for bx, by, bsize in butterflies:
        # Generate butterfly points
        x, y = _generate_butterfly(points=1000, size=bsize, offset_x=bx, offset_y=by)
        
        # Create line segments with varying widths
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Vary line width with sine wave
        n_points = len(x)
        t = np.linspace(0, 2 * np.pi, n_points)
        
        # Base width with sine variation for natural tapering
        base_width = 0.5
        width_variation = 3
        width = base_width + width_variation * (np.sin(t * 2) + 1) / 2  # 0 to 1 range
        
        # Create color gradient
        color_values = np.sin(t * 2)  # Vary color with sine wave
        cmap = LinearSegmentedColormap.from_list('butterfly', [_random_rgb(), _random_rgb(), _random_rgb()])
        norm = plt.Normalize(-1, 1)
        
        # Create line collection with varying widths and colors
        lc = LineCollection(segments, 
                          linewidths=width, 
                          colors=cmap(norm(color_values)),
                          capstyle='round',
                          alpha=0.9,
                          zorder=2)
        
        # Add filled version of the butterfly
        ax.fill(x, y, color=_random_rgb(), alpha=0.7, zorder=1, linewidth=0)
        ax.add_collection(lc)
        
    ax.set_facecolor('black')
    ax.set_xlim(-canvas_size/2, canvas_size/2)
    ax.set_ylim(-canvas_size/2, canvas_size/2)
    plt.axis('off')
    
    if save_path:
        # Ensure the figure is the exact size we want
        fig.set_size_inches(width_px/dpi, height_px/dpi)
        plt.savefig(save_path, 
                   dpi=dpi, 
                   bbox_inches='tight', 
                   pad_inches=0,
                   facecolor='black', 
                   edgecolor='none')
        print(f"Butterfly image saved to {save_path} ({width_px}x{height_px}px)")

def _generate_butterfly_key(width=256, height=None, canvas_size=256, output_path=None):
    """Generate a butterfly key image with the specified dimensions.
    
    Args:
        width (int): Width of the output image in pixels
        height (int, optional): Height of the output image in pixels. 
                              If None, uses the same as width (square image).
        canvas_size (int): Internal coordinate system size for butterfly generation.
                         Larger values result in more detailed butterflies.
        output_path (str, optional): Path to save the image. If None, a temporary file is created.
        
    Returns:
        str: Path to the generated image file
    """
    if height is None:
        height = width  # Default to square image if height not specified
        
    # Determine the larger dimension for butterfly scaling
    max_dim = max(width, height)
    
    # Adjust butterfly sizes based on the image dimensions
    butterflies = []
    butterflies.extend(_generate_butterflies(n=5, size_range=(0.6, 1.5), canvas_size=canvas_size))
    butterflies.extend(_generate_butterflies(n=8, size_range=(0.3, 0.7), canvas_size=canvas_size))
    butterflies.extend(_generate_butterflies(n=12, size_range=(0.1, 0.4), canvas_size=canvas_size))
    
    # Use a temporary file if no output path is provided
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    # Generate the image with the specified dimensions
    _plot_butterflies(
        butterflies, 
        save_path=output_path, 
        width_px=width,
        height_px=height,
        canvas_size=canvas_size
    )
    
    return output_path


def _generate_qr_positions(n=5, size_range=(0.1, 0.3), canvas_size=1000, min_padding=0.05):
    """Generate random positions and sizes for QR codes.
    
    Args:
        n: Number of QR codes to generate
        size_range: Tuple of (min_size, max_size) as a fraction of canvas size
        canvas_size: Size of the canvas
        min_padding: Minimum padding from edges
        
    Returns:
        List of (x, y, size) tuples for each QR code
    """
    qr_positions = []
    min_size = int(canvas_size * size_range[0])
    max_size = int(canvas_size * size_range[1])
    
    for _ in range(n):
        size = random.randint(min_size, max_size)
        padding = int(size * min_padding)
        x = random.randint(padding, canvas_size - size - padding)
        y = random.randint(padding, canvas_size - size - padding)
        qr_positions.append((x, y, size))
    
    return qr_positions

def _generate_qr_key(width=256, height=None, data="aposematic qr key", canvas_size=256, output_path=None):
    """Generate a QR key image with multiple scattered QR codes on a noise background."""
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    from qrcode.image.styledpil import StyledPilImage
    from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
    
    if height is None:
        height = width  # Default to square image if height not specified
        
    # Create a noise background using matplotlib
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Add noise background
    _generate_background(ax, size=canvas_size,
                        colors=[_random_rgb(), _random_rgb(), _random_rgb()],
                        dark_theme=True)
    
    # Convert figure to numpy array
    fig.canvas.draw()
    
    # Get the image as an array
    if hasattr(fig.canvas, 'tostring_rgb'):
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_data = img_data[..., :3]  # Drop alpha channel
    
    plt.close(fig)
    
    # Convert numpy array to PIL Image
    background = Image.fromarray(img_data).convert('RGBA')
    result = Image.new('RGBA', background.size)
    result.paste(background, (0, 0))
    
    # Generate multiple QR codes at random positions
    qr_positions = _generate_qr_positions(
        n=random.randint(6, 10),  # Number of QR codes
        size_range=(0.1, 0.4),   # Size range as fraction of canvas
        canvas_size=canvas_size
    )
    
    # Generate a higher resolution version of the background
    scale_factor = 4  # Scale up for better QR code quality
    large_canvas = background.copy().resize((canvas_size * scale_factor, canvas_size * scale_factor), 
                                          Image.Resampling.NEAREST)
    
    for x, y, size in qr_positions:
        # Generate a unique data string for each QR code
        qr_data = f"{data}_{random.getrandbits(32):x}"
        
        # Generate the QR code at a higher resolution
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=4,  # Increased box size for better quality
            border=1,    # Original border size
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        # Create the QR code with random color and magenta background
        qr_color = _random_rgb_int()
        
        # Create a temporary image with magenta background
        qr_img = qr.make_image(
            fill_color=qr_color,
            back_color="magenta"
        )
        qr_img = qr_img.convert('RGBA')
        
        # Make magenta transparent
        data = qr_img.getdata()
        new_data = []
        for item in data:
            # Change all magenta (or near-magenta) pixels to be transparent
            if item[0] > 200 and item[1] < 50 and item[2] > 200:  # Magenta threshold
                new_data.append((0, 0, 0, 0))  # Transparent
            else:
                new_data.append(item)  # Keep original color
        
        qr_img.putdata(new_data)
        
        # Calculate position and size at high resolution
        x_hr, y_hr = int(x * scale_factor), int(y * scale_factor)
        size_hr = int(size * scale_factor)
        
        # Resize using NEAREST to maintain sharp edges
        qr_img = qr_img.resize((size_hr, size_hr), Image.Resampling.NEAREST)
        
        # Create a new image to handle transparency correctly
        qr_final = Image.new('RGBA', large_canvas.size, (0, 0, 0, 0))
        qr_final.paste(qr_img, (x_hr, y_hr))
        
        # Composite onto the high-res canvas
        large_canvas = Image.alpha_composite(
            large_canvas.convert('RGBA'),
            qr_final
        )
    
    # Convert back to RGB and scale down
    result = large_canvas.convert('RGB')
    result = result.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    
    # Save the result
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    result.save(output_path, 'PNG')
    print(f"QR code with noise background saved to {output_path} ({width}x{height}px)")
    return output_path

def _permute(x):
    return _permutation[x]

def _inverse_permute(x):
    return _inv_permutation[x]

OPS = {
    # Addition/Subtraction with key - perfectly reversible
    '+': lambda a, b: (a.astype(np.int32) + b.astype(np.int32)) % 256,
    '-': lambda a, b: (a.astype(np.int32) - b.astype(np.int32)) % 256,
    
    # XOR with key - perfectly reversible
    '^': lambda a, b: a.astype(np.int32) ^ b.astype(np.int32),
    
    # Bit rotation by 1 - perfectly reversible
    '>': lambda a, b: ((a.astype(np.int32) >> 1) | ((a & 1) << 7)) & 0xFF,
    '<': lambda a, b: ((a.astype(np.int32) << 1) | ((a >> 7) & 1)) & 0xFF,
    
    # Permutation - perfectly reversible
    'p': lambda a, b: _permute(a),
    'P': lambda a, b: _inverse_permute(a),
    
    # Add with a constant (using LSB of key for variation)
    'a': lambda a, b: (a.astype(np.int32) + (b[0] & 0x0F) + 1) % 256,
    'A': lambda a, b: (a.astype(np.int32) - (b[0] & 0x0F) - 1) % 256,
}

INV_OPS = {
    '+': '-',   # a+b → a = (a+b) - b
    '-': '+',   # a-b → a = (a-b) + b
    '^': '^',   # XOR is its own inverse
    '>': '<',   # Rotate right → rotate left
    '<': '>',   # Rotate left → rotate right
    'p': 'P',   # Permutation → inverse permutation
    'P': 'p',   # Inverse permutation → permutation
    'a': 'A',   # Add constant → subtract constant
    'A': 'a',   # Subtract constant → add constant
}

def scramble(original_img_path, key_img_path=None, op_string="-^+", scramble_mode=SCRAMBLE_MODE.NONE, output_path=None):
    """
    Scramble original using key image and repeating op_string.
    
    Returns:
        dict: Dictionary containing:
            - 'scrambled_path': Path to the scrambled image
            - 'key_path': Path to the key image
    """
    img = cv2.imread(original_img_path)
    if img is None:
        raise ValueError(f"Could not load image: {original_img_path}")
    
    # Generate a key if none provided
    if key_img_path is None:
        if scramble_mode == SCRAMBLE_MODE.NONE:
            scramble_mode = SCRAMBLE_MODE.BUTTERFLY
            
        if scramble_mode == SCRAMBLE_MODE.BUTTERFLY:
            key_path = _generate_butterfly_key(
                width=img.shape[1],
                height=img.shape[0],
                canvas_size=max(img.shape[0], img.shape[1])
            )
        elif scramble_mode == SCRAMBLE_MODE.QR:
            key_path = _generate_qr_key(
                width=img.shape[1],
                height=img.shape[0],
                data=f"Aiposematic Key {os.urandom(8).hex()}",
                canvas_size=max(img.shape[0], img.shape[1])
            )
        else:
            raise ValueError(f"Unsupported scramble mode: {scramble_mode}")
            
        key = cv2.imread(key_path)
    else:
        # Use provided key image
        key = cv2.imread(key_img_path)
        if key is None:
            raise ValueError(f"Could not load key image: {key_img_path}")
        key_path = key_img_path
    
    # Resize key to match input image if needed
    if img.shape[:2] != key.shape[:2]:
        key = cv2.resize(key, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Process the image with the key
    h, w, c = img.shape
    locked = np.zeros_like(img)
    
    # Flatten and cycle operations
    pixels = img.reshape(-1, c)
    key_pixels = key.reshape(-1, c)
    ops_cycle = itertools.cycle(op_string)
    
    for i in range(len(pixels)):
        op = next(ops_cycle)
        func = OPS[op]
        locked_flat = func(pixels[i], key_pixels[i])
        locked.reshape(-1, c)[i] = locked_flat
    
    locked = locked.astype(np.uint8)
    
    # Handle output path
    if output_path is None:
        import tempfile
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    # Save the scrambled image
    cv2.imwrite(output_path, locked)
    print(f"Scrambled image saved to: {output_path}")
    
    return {
        'scrambled_path': output_path,
        'key_path': key_path
    }

def recover(locked_img_path, key_img_path, op_string="-^+", output_path=None):
    """
    Recover original using same key and op_string.
    """
    locked = cv2.imread(locked_img_path)
    if locked is None:
        raise ValueError(f"Could not load locked image: {locked_img_path}")
    
    key = cv2.imread(key_img_path)
    if key is None:
        raise ValueError(f"Could not load key image: {key_img_path}")
    
    # Resize key to match locked image if needed
    if locked.shape[:2] != key.shape[:2]:
        key = cv2.resize(key, (locked.shape[1], locked.shape[0]), 
                        interpolation=cv2.INTER_NEAREST)
    
    h, w, c = locked.shape
    recovered = np.zeros_like(locked)
    
    # Flatten the arrays for processing
    pixels = locked.reshape(-1, c)
    key_pixels = key.reshape(-1, c)
    
    # Process operations in the same order as scramble, but with inverted operations
    ops_cycle = itertools.cycle(op_string)
    
    for i in range(len(pixels)):
        op = next(ops_cycle)
        inv_op = INV_OPS[op]  # Get inverse operation
        func = OPS[inv_op]    # Get the function for inverse operation
        recovered_flat = func(pixels[i], key_pixels[i])
        recovered.reshape(-1, c)[i] = recovered_flat
    
    recovered = recovered.astype(np.uint8)
    
    # Use a temporary file if no output path is provided
    if output_path is None:
        import tempfile
        output_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    cv2.imwrite(output_path, recovered)
    print(f"Recovered image saved: {output_path}")
    return output_path

def get_seeded_random(seed):
    """Create a deterministic random number generator from a seed."""
    import hashlib
    # Use a hash of the seed to ensure good distribution
    if isinstance(seed, str):
        seed = seed.encode('utf-8')
    if not isinstance(seed, bytes):
        seed = str(seed).encode('utf-8')
    seed_hash = int(hashlib.sha256(seed).hexdigest(), 16)
    import random
    random.seed(seed_hash)
    return random

def shuffle_image_pixels(image_path, seed, output_path=None):
    """Shuffle the pixels of an image using a seed."""
    from PIL import Image
    import numpy as np
    
    # Open the image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    arr = np.array(img)
    height, width = arr.shape[:2]
    total_pixels = height * width
    
    # Flatten the array
    flat = arr.reshape(-1, 3)
    
    # Generate deterministic shuffle indices
    rng = get_seeded_random(seed)
    indices = np.arange(total_pixels)
    rng.shuffle(indices)
    
    # Apply the shuffle
    shuffled = flat[indices]
    
    # Reshape back to original dimensions
    result = shuffled.reshape(height, width, 3)
    
    # Save or return the result
    if output_path:
        result_img = Image.fromarray(result)
        result_img.save(output_path)
        return output_path
    return Image.fromarray(result)

def unshuffle_image_pixels(shuffled_img_path, seed, output_path=None):
    """Unshuffle an image that was shuffled with shuffle_image_pixels."""
    from PIL import Image
    import numpy as np
    
    # Open the image
    img = Image.open(shuffled_img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    arr = np.array(img)
    height, width = arr.shape[:2]
    total_pixels = height * width
    
    # Flatten the array
    flat = arr.reshape(-1, 3)
    
    # Generate the same shuffle indices
    rng = get_seeded_random(seed)
    indices = np.arange(total_pixels)
    rng.shuffle(indices)
    
    # Create inverse mapping
    inverse_indices = np.argsort(indices)
    
    # Apply inverse shuffle
    unshuffled = flat[inverse_indices]
    
    # Reshape and save/return
    result = unshuffled.reshape(height, width, 3)
    if output_path:
        result_img = Image.fromarray(result)
        result_img.save(output_path)
        return output_path
    return Image.fromarray(result)

def add_ai_deterrent_features(image_path, output_path):
    """
    Add AI-deterrent features to an image.
    This includes high-frequency noise and edge-based modifications.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # 1. Add high-frequency noise (barely visible to humans)
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img = cv2.addWeighted(img, 0.95, noise, 0.05, 0)
    
    # 2. Add subtle adversarial patterns in edge regions
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    img[edges > 0] = cv2.addWeighted(img[edges > 0], 0.9, 
                                    np.random.randint(0, 50, img[edges > 0].shape, np.uint8), 
                                    0.1, 0)
    
    cv2.imwrite(output_path, img)
    return output_path

def new_aposematic_img(original_img_path, op_string='-^+', scramble_mode=SCRAMBLE_MODE.BUTTERFLY, output_path=None):
    """
    Create a new aposematic image using pixel shuffling encryption.
    """
    # Create a temporary file for the AI-deterred image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Step 0: Apply AI deterrent features to the original image
        deterred_path = add_ai_deterrent_features(original_img_path, temp_path)
        
        # Step 1: Scramble the original image with AI deterrents
        result = scramble(
            original_img_path=deterred_path,
            key_img_path=None,  # Generate a new key
            op_string=op_string,
            scramble_mode=scramble_mode,
            output_path=None  # Will generate a temp path
        )
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise  # Re-raise the exception
    
    # Step 2: Generate a secure random seed
    seed = secrets.token_hex(16)  # 128-bit secure random seed
    
    # Step 3: Shuffle the key image
    shuffled_key_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    shuffle_image_pixels(result['key_path'], seed, shuffled_key_path)
    
    # Step 4: Hide the shuffled key in the scrambled image
    final_image_path = output_path or tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    _steganography_encode(result['scrambled_path'], shuffled_key_path, final_image_path)
    
    # Clean up temporary files
    temp_files = [result['scrambled_path'], result['key_path'], shuffled_key_path, temp_path]
    for path in temp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {path}: {e}")
    
    return {
        'img_path': final_image_path,
        'cipher_key': seed  # Return the seed as the encryption key
    }

def recover_aposematic_img(aposematic_img_path, cipher_key, op_string='-^+', output_path=None):
    """
    Recover the original image from an aposematic image using pixel unshuffling.
    """
    try:
        # Step 1: Extract the shuffled key image
        shuffled_key_path = _steganography_decode(aposematic_img_path)
        
        try:
            # Step 2: Unshuffle the key image
            unshuffled_key_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
            unshuffle_image_pixels(shuffled_key_path, cipher_key, unshuffled_key_path)
            
            try:
                # Step 3: Recover the original image
                recovered_path = recover(
                    locked_img_path=aposematic_img_path,
                    key_img_path=unshuffled_key_path,
                    op_string=op_string,
                    output_path=output_path
                )
                return recovered_path
            finally:
                # Clean up the unshuffled key
                if os.path.exists(unshuffled_key_path):
                    os.unlink(unshuffled_key_path)
        finally:
            # Clean up the extracted key
            if os.path.exists(shuffled_key_path):
                os.unlink(shuffled_key_path)
                
    except Exception as e:
        print(f"Error recovering aposematic image: {str(e)}")
        raise

        