import os
import cv2
import numpy as np
from aiposematic import scramble, recover, SCRAMBLE_MODE

def display_image(window_name, image_path, wait_time=4000):
    """Display an image in a window and wait for a short time or key press."""
    img = cv2.imread(image_path)
    if img is not None:
        # Create a resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Resize if the image is too large for the screen
        height, width = img.shape[:2]
        max_size = 800
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # Show the image
        cv2.imshow(window_name, img)
        print(f"Displaying {window_name} for {wait_time//1000} seconds (press any key to continue)...")
        
        # Wait for key press or timeout
        key = cv2.waitKey(wait_time) & 0xFF
        if key != 255:  # If any key was pressed
            cv2.waitKey(0)  # Wait until another key is pressed
        
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Small delay to ensure window is closed
    else:
        print(f"Warning: Could not display {image_path}")

def test_scramble_recover(input_image, mode, op_sequence="-^+"):
    """Test scrambling and recovering an image with the given mode."""
    print(f"\n{'='*50}")
    print(f"Testing with {mode.name} key generation")
    print(f"{'='*50}")
    
    try:
        # Test scrambling with auto-generated key
        print(f"\nScrambling with {mode.name} key...")
        result = scramble(
            input_image,
            key_img_path=None,
            op_string=op_sequence,
            scramble_mode=mode
        )
        
        scrambled_path = result['scrambled_path']
        key_path = result['key_path']
        
        print(f"Generated key: {key_path}")
        print(f"Scrambled image saved to: {scrambled_path}")
        
        # Display the original, key, and scrambled images
        display_image("Original Image", input_image)
        display_image(f"{mode.name} Key", key_path)
        display_image("Scrambled Image", scrambled_path)
        
        # Test recovery with correct key
        print("\nRecovering with correct key...")
        recover(scrambled_path, key_path, op_string=op_sequence, output_path="recovered.png")
        display_image("Recovered Image (Correct Key)", "recovered.png")
        
        # Test recovery with wrong key
        print("\nTesting with wrong key...")
        wrong_key = cv2.imread(key_path)
        if wrong_key is not None:
            wrong_key = 255 - wrong_key  # Invert colors to create wrong key
            cv2.imwrite("wrong_key.png", wrong_key)
            
            try:
                recover(scrambled_path, "wrong_key.png", op_string=op_sequence, output_path="failed_recovery.png")
                display_image("Recovered Image (Wrong Key)", "failed_recovery.png")
                print("WARNING: Recovery succeeded with wrong key! This may indicate a security issue.")
            except Exception as e:
                print(f"Expected error with wrong key: {str(e)}")
        
    finally:
        # Clean up
        for path in [scrambled_path, key_path, 'recovered.png', 'wrong_key.png', 'failed_recovery.png']:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Warning: Could not delete {path}: {e}")

def main():
    # Input image
    input_image = "original.png"
    
    # Check if the image exists
    if not os.path.exists(input_image):
        raise FileNotFoundError(
            f"Test image '{input_image}' not found in the repository. "
            "Please ensure it exists in the current directory."
        )
    
    print(f"Using test image: {input_image}")
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f"Could not read image: {input_image}")
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
    
    # Test with different modes
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        test_scramble_recover(input_image, mode)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()