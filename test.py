import os
import numpy as np
import cv2
from aiposematic import scramble, recover

def calculate_entropy(image):
    """Calculate the entropy of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist[hist != 0] / (image.shape[0] * image.shape[1])
    return -np.sum(hist * np.log2(hist))

def analyze_image_encryption(original_path, locked_path, recovered_path):
    """Analyze the effectiveness of the image encryption."""
    # Load images
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    locked = cv2.imread(locked_path, cv2.IMREAD_GRAYSCALE)
    recovered = cv2.imread(recovered_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Entropy analysis
    orig_entropy = calculate_entropy(original)
    locked_entropy = calculate_entropy(locked)
    
    # 2. Histogram comparison
    orig_hist = cv2.calcHist([original], [0], None, [256], [0, 256])
    locked_hist = cv2.calcHist([locked], [0], None, [256], [0, 256])
    hist_corr = np.corrcoef(orig_hist.flatten(), locked_hist.flatten())[0, 1]
    
    # 3. SSIM (Structural Similarity)
    ssim_score = ssim(original, locked, data_range=locked.max() - locked.min())
    
    # 4. Edge detection comparison
    edges_orig = cv2.Canny(original, 100, 200)
    edges_locked = cv2.Canny(locked, 100, 200)
    edge_similarity = np.sum(edges_orig == edges_locked) / edges_orig.size
    
    # 5. Correlation coefficient
    corr_coeff = np.corrcoef(original.flatten(), locked.flatten())[0, 1]
    
    # 6. Pixel difference
    diff = cv2.absdiff(original, recovered)
    diff_pixels = np.count_nonzero(diff)
    total_pixels = original.shape[0] * original.shape[1]
    
    print("\n--- Encryption Effectiveness Analysis ---")
    print(f"Original Entropy: {orig_entropy:.4f} (higher is better, max ~8 for 8-bit grayscale)")
    print(f"Locked Entropy: {locked_entropy:.4f} (should be close to 8 for good encryption)")
    print(f"Histogram Correlation: {hist_corr:.6f} (closer to 0 is better)")
    print(f"SSIM Score: {ssim_score:.6f} (closer to 0 is better)")
    print(f"Edge Similarity: {edge_similarity:.6f} (closer to 0 is better)")
    print(f"Correlation Coefficient: {corr_coeff:.6f} (closer to 0 is better)")
    print(f"Pixels changed in recovery: {diff_pixels}/{total_pixels} (should be 0 for perfect recovery)")
    
    return {
        'original_entropy': orig_entropy,
        'locked_entropy': locked_entropy,
        'histogram_correlation': hist_corr,
        'ssim': ssim_score,
        'edge_similarity': edge_similarity,
        'correlation': corr_coeff,
        'pixels_changed': diff_pixels,
        'total_pixels': total_pixels
    }

def make_key(size=(256, 256), filename="key.png"):
    """Generate a random key image if it doesn't exist."""
    if not os.path.exists(filename):
        key = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        cv2.imwrite(filename, key)
        print(f"Created new key: {filename}")
    else:
        key = cv2.imread(filename)
        print(f"Using existing key: {filename}")
    return key

def load_image(filename):
    """Load an image from file, ensuring it's a PNG."""
    if not filename.lower().endswith('.png'):
        raise ValueError(f"Only PNG files are supported: {filename}")
        
    if not os.path.exists(filename):
        # If the file doesn't exist, try to create a test image if it's the original or key
        if 'original' in filename.lower():
            print(f"Creating test image: {filename}")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(img, 'Test Image', (100, 256), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.imwrite(filename, img)
        elif 'key' in filename.lower():
            print(f"Creating random key: {filename}")
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(filename, img)
        else:
            raise FileNotFoundError(f"Image not found: {filename}")
    else:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {filename}")
    
    print(f"Loaded image: {filename} ({img.shape[1]}x{img.shape[0]})")
    return img

# Input and output filenames (only PNG format)
input_image = "original.png"
key_file = "key.png"
output_locked = "original_locked.png"
output_recovered = "original_recovered.png"
output_failed = "original_failed.png"

# Load the input image and key
img = load_image(input_image)
key = load_image(key_file)

# Resize key to match image if needed
if key.shape != img.shape[:2]:
    key = cv2.resize(key, (img.shape[1], img.shape[0]))
    print(f"Resized key to match image dimensions: {img.shape[1]}x{img.shape[0]}")

# Operation sequence for obfuscation
#op_sequence = "+^>p<a"  # Add, XOR, rotate right, permute, rotate left, add with key
op_sequence = "^"

print("\nObfuscating image...")
scramble(input_image, key_file, op_string=op_sequence, output_path=output_locked)
print(f"Obfuscated image saved as: {output_locked}")

print("\nRecovering image with correct key...")
recover(output_locked, key_file, op_string=op_sequence, output_path=output_recovered)
print(f"Recovered image saved as: {output_recovered}")

print("\nTrying with wrong key...")
# Create a wrong key by shifting and inverting the original key
wrong_key = np.roll(key, 1, axis=0)
wrong_key = 255 - wrong_key  # Invert colors for more obvious difference
wrong_key_path = "wrong_key.png"
cv2.imwrite(wrong_key_path, wrong_key)
print(f"Created wrong key: {wrong_key_path}")

recover(output_locked, wrong_key_path, op_string=op_sequence, output_path=output_failed)
print(f"Failed recovery attempt saved as: {output_failed}")

print("\nProcess completed successfully!")