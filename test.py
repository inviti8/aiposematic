import cv2
import os
import tempfile
from aiposematic import (
    SCRAMBLE_MODE,
    scramble,
    recover,
    new_aposematic_img,
    recover_aposematic_img
)

def display_image(title, image_path, display_time=4000):
    """Display an image using OpenCV."""
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imshow(title, img)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()
    else:
        print(f"Warning: Could not display {image_path}")

def test_scramble_recover(input_image, mode, op_sequence="-^+"):
    """Test scrambling and recovering an image with the given mode."""
    print(f"\n{'='*50}")
    print(f"Testing with {mode.name} key generation")
    print("="*50 + "\n")

    # Scramble the image
    print("Scrambling with", mode.name, "key...")
    result = scramble(
        original_img_path=input_image,
        key_img_path=None,  # Generate a new key
        op_string=op_sequence,
        scramble_mode=mode,
        output_path=None  # Will generate a temp path
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
    recovered_path = recover(
        locked_img_path=scrambled_path,
        key_img_path=key_path,
        op_string=op_sequence
    )
    print(f"Recovered image saved: {recovered_path}")
    display_image("Recovered Image (Correct Key)", recovered_path)

    # Test recovery with wrong key (should fail)
    print("\nTesting with wrong key...")
    try:
        wrong_recovered = recover(
            locked_img_path=scrambled_path,
            key_img_path=input_image,  # Using original image as wrong key
            op_string=op_sequence
        )
        print(f"Recovered image saved: {wrong_recovered}")
        display_image("Recovered Image (Wrong Key)", wrong_recovered)
        print("WARNING: Recovery succeeded with wrong key! This may indicate a security issue.")
    except Exception as e:
        print(f"Expected error with wrong key (this is good): {str(e)}")

    # Clean up
    for path in [scrambled_path, key_path, recovered_path, 
                'recovered.png', 'wrong_key.png', 'failed_recovery.png']:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

def test_aposematic_workflow(input_image, mode=SCRAMBLE_MODE.BUTTERFLY):
    """Test the complete aposematic image workflow."""
    print("\n" + "="*50)
    print(f"Testing Aposematic Image Workflow with {mode.name} Key")
    print("="*50)
    
    # 1. Create a new aposematic image
    print("\nCreating aposematic image...")
    result = new_aposematic_img(
        original_img_path=input_image,
        op_string="-^+",
        scramble_mode=mode
    )
    aposematic_path = result['img_path']
    cipher_key = result['cipher_key']
    
    print(f"\nAposematic image created at: {aposematic_path}")
    print(f"Cipher key (truncated): {cipher_key[:30]}...")
    
    # Display the aposematic image
    display_image(f"Aposematic Image ({mode.name} Key)", aposematic_path)
    
    # 2. Recover the original image
    print("\nRecovering original image...")
    recovered_path = recover_aposematic_img(
        aposematic_img_path=aposematic_path,
        cipher_key=cipher_key,
        op_string="-^+"
    )
    print(f"Recovered image saved to: {recovered_path}")
    
    # Display the recovered image
    display_image(f"Recovered Image ({mode.name} Key)", recovered_path)
    
    # Clean up
    for path in [aposematic_path, recovered_path]:
        try:
            if os.path.exists(path):
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
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}\n")
    
    # Test basic scramble/recover for each mode
    print("\n" + "="*50)
    print("Testing Basic Scramble/Recovery")
    print("="*50)
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        test_scramble_recover(input_image, mode)
    
    # Test aposematic workflow for each mode
    print("\n" + "="*50)
    print("Testing Aposematic Workflow")
    print("="*50)
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        test_aposematic_workflow(input_image, mode)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()