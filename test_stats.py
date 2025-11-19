import os
import cv2
import numpy as np
from aiposematic import (
    SCRAMBLE_MODE,
    scramble,
    new_aposematic_img
)
from statistical_analysis import ImageAnalyzer, compare_images

def run_analysis(original_path, output_dir):
    """Run statistical analysis on original and scrambled images."""
    print(f"\n{'='*50}")
    print(f"Running Statistical Analysis")
    print(f"Original image: {original_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze the original image
    print("Analyzing original image...")
    original_analyzer = ImageAnalyzer(image_path=original_path)
    original_results = original_analyzer.run_all_analyses(
        save_dir=os.path.join(output_dir, "original")
    )
    
    # 2. Create scrambled versions
    print("\nCreating scrambled images...")
    scrambles = {}
    for mode in [SCRAMBLE_MODE.BUTTERFLY, SCRAMBLE_MODE.QR]:
        # Create aposematic image (scrambled with key)
        print(f"\nCreating {mode.name} scrambled image...")
        result = new_aposematic_img(
            original_img_path=original_path,
            op_string="-^+",
            scramble_mode=mode
        )
        
        # Save paths and analyzer
        scrambles[mode.name.lower()] = {
            'path': result['img_path'],
            'cipher_key': result['cipher_key']
        }
        
        # Analyze the scrambled image
        print(f"Analyzing {mode.name} scrambled image...")
        scrambler = ImageAnalyzer(image_path=result['img_path'])
        scrambler.run_all_analyses(
            original_image_path=original_path,
            save_dir=os.path.join(output_dir, f"scrambled_{mode.name.lower()}")
        )
        
        # Compare original with scrambled
        print(f"Comparing original with {mode.name} scrambled...")
        compare_results = compare_images(
            original_path,
            result['img_path'],
            save_dir=os.path.join(output_dir, f"comparison_{mode.name.lower()}")
        )
    
    print("\nAnalysis complete! Results saved to:", os.path.abspath(output_dir))
    print("\nSummary of comparisons:")
    
    # Print summary of comparisons
    for mode in scrambles:
        comp_dir = os.path.join(output_dir, f"comparison_{mode}")
        try:
            with open(os.path.join(comp_dir, 'comparison_summary.txt'), 'r') as f:
                print(f"\n{mode.upper()} Scrambling:")
                print("-" * (len(mode) + 12))
                print(f.read())
        except Exception as e:
            print(f"Could not load comparison summary for {mode}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run statistical analysis on original and scrambled images')
    parser.add_argument('--input', '-i', default='original.png',
                      help='Path to the original image (default: original.png)')
    parser.add_argument('--output', '-o', default='analysis_results',
                      help='Output directory for analysis results (default: analysis_results)')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)
    
    run_analysis(args.input, args.output)