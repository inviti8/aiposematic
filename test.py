import cv2
import os
import numpy as np
import tempfile
from aiposematic import (
    SCRAMBLE_MODE,
    scramble,
    recover,
    new_aposematic_img,
    recover_aposematic_img,
    derive_cipher_key,
    OP_INTENSITY,
)

try:
    from hvym_stellar import Stellar25519KeyPair
    from stellar_sdk import Keypair as StellarKeypair
    _HAS_STELLAR = True
except ImportError:
    _HAS_STELLAR = False

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
    """Test v1.0 scramble/recover with key image + cipher_key."""
    print(f"\n{'='*50}")
    print(f"Testing v1.0 scramble/recover with {mode.name} key")
    print("="*50 + "\n")

    # Scramble the image
    print("Scrambling with", mode.name, "key...")
    result = scramble(
        original_img_path=input_image,
        cipher_key=None,  # Generate a new cipher_key
        op_string=op_sequence,
        scramble_mode=mode,
        output_path=None
    )

    scrambled_path = result['scrambled_path']
    cipher_key = result['cipher_key']
    key_path = result['key_path']
    print(f"Cipher key: {cipher_key}")
    print(f"Key image: {key_path}")
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
        cipher_key=cipher_key,
        op_string=op_sequence
    )
    print(f"Recovered image saved: {recovered_path}")
    display_image("Recovered Image (Correct Key)", recovered_path)

    # Verify pixel-perfect recovery
    original_img = cv2.imread(input_image)
    recovered_img = cv2.imread(recovered_path)
    if np.array_equal(original_img, recovered_img):
        print("PASS: Pixel-perfect recovery!")
    else:
        diff = np.abs(original_img.astype(int) - recovered_img.astype(int))
        print(f"WARN: Recovery not pixel-perfect. Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}")

    # Test recovery with wrong key (should produce garbage)
    print("\nTesting with wrong cipher_key...")
    wrong_key = "00" * 16
    wrong_recovered = recover(
        locked_img_path=scrambled_path,
        key_img_path=key_path,
        cipher_key=wrong_key,
        op_string=op_sequence
    )
    wrong_img = cv2.imread(wrong_recovered)
    if not np.array_equal(original_img, wrong_img):
        print("PASS: Wrong cipher_key produces different output.")
    else:
        print("FAIL: Wrong cipher_key produced identical output!")

    # Clean up
    for path in [scrambled_path, key_path, recovered_path, wrong_recovered]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

def test_aposematic_workflow(input_image, mode=SCRAMBLE_MODE.BUTTERFLY):
    """Test the complete aposematic image workflow."""
    print("\n" + "="*50)
    print(f"Testing Aposematic Workflow with {mode.name} Key")
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
    print(f"Cipher key: {cipher_key}")

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

def test_wrong_key_v1(input_image):
    """Wrong cipher_key produces garbage, not partial recovery."""
    print("\n" + "="*50)
    print("Testing v1.0: Wrong Key Produces Garbage")
    print("="*50)

    result = new_aposematic_img(
        original_img_path=input_image,
        op_string="-^+",
    )

    # Recover with wrong key
    wrong_key = "ff" * 16
    wrong_recovered = recover_aposematic_img(
        aposematic_img_path=result['img_path'],
        cipher_key=wrong_key,
        op_string="-^+"
    )

    # Recover with correct key
    correct_recovered = recover_aposematic_img(
        aposematic_img_path=result['img_path'],
        cipher_key=result['cipher_key'],
        op_string="-^+"
    )

    wrong_img = cv2.imread(wrong_recovered)
    correct_img = cv2.imread(correct_recovered)

    if not np.array_equal(wrong_img, correct_img):
        diff = np.abs(wrong_img.astype(int) - correct_img.astype(int))
        print(f"PASS: Wrong key output differs. Mean diff: {diff.mean():.1f}")
    else:
        print("FAIL: Wrong key produced same output as correct key!")

    for path in [result['img_path'], wrong_recovered, correct_recovered]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except:
            pass

def test_intensity_variation(input_image):
    """Demonstrate that different op_strings produce different disruption levels."""
    print("\n" + "="*50)
    print("Testing v1.1: Per-Op Intensity Variation")
    print("="*50)

    op_strings = [">", "a", "p", "-^+", "-^+p>a<PA"]
    for op_str in op_strings:
        result = scramble(
            original_img_path=input_image,
            op_string=op_str,
        )
        scrambled = cv2.imread(result['scrambled_path'])
        original = cv2.imread(input_image)
        diff = np.abs(scrambled.astype(float) - original.astype(float)).mean()
        print(f"  op_string='{op_str}' -> mean pixel diff = {diff:.2f}")

        # Verify roundtrip
        recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            cipher_key=result['cipher_key'],
            op_string=op_str,
        )
        rec_img = cv2.imread(recovered)
        assert np.array_equal(original, rec_img), f"Roundtrip failed for op_string='{op_str}'"

        for p in [result['scrambled_path'], result['key_path'], recovered]:
            if os.path.exists(p):
                os.unlink(p)

    print("PASS: All intensity roundtrips pixel-perfect!")


def test_stellar_workflow(input_image):
    """Test scramble/recover using Stellar keypair for key derivation."""
    if not _HAS_STELLAR:
        print("\n" + "="*50)
        print("SKIP: Stellar workflow test (hvym-stellar not installed)")
        print("="*50)
        return

    print("\n" + "="*50)
    print("Testing v1.1: Stellar Key Integration")
    print("="*50)

    kp = Stellar25519KeyPair(StellarKeypair.random())
    print(f"  Public key: {kp.public_key()[:16]}...")

    # Scramble using Stellar keypair
    result = new_aposematic_img(
        original_img_path=input_image,
        op_string="-^+",
        stellar_keypair=kp,
    )
    print(f"  Derived cipher_key: {result['cipher_key']}")
    display_image("Stellar Aposematic", result['img_path'])

    # Recover using same Stellar keypair
    recovered = recover_aposematic_img(
        aposematic_img_path=result['img_path'],
        op_string="-^+",
        stellar_keypair=kp,
    )
    display_image("Stellar Recovered", recovered)

    # Clean up
    for p in [result['img_path'], recovered]:
        if os.path.exists(p):
            os.unlink(p)

    print("PASS: Stellar self-recovery workflow!")


def main():
    input_image = "original.png"

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

    # Test basic scramble/recover
    print("\n" + "="*50)
    print("Testing Basic Scramble/Recovery")
    print("="*50)
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        test_scramble_recover(input_image, mode)

    # Test aposematic workflow
    print("\n" + "="*50)
    print("Testing Aposematic Workflow")
    print("="*50)
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        test_aposematic_workflow(input_image, mode)

    # v1.0 wrong key test
    test_wrong_key_v1(input_image)

    # v1.1 intensity variation test
    test_intensity_variation(input_image)

    # v1.1 Stellar integration test
    test_stellar_workflow(input_image)

    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
