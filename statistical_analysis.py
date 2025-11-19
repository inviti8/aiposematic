import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
from skimage import util
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Dict, List, Optional
import os

class ImageAnalyzer:
    """
    A class to perform statistical analysis on images to evaluate their AI resistance properties.
    """
    
    def __init__(self, image_path: str = None, image_array: np.ndarray = None):
        """
        Initialize the ImageAnalyzer with either a file path or a numpy array.
        
        Args:
            image_path: Path to the image file
            image_array: Numpy array containing the image data (BGR format if color)
        """
        if image_path:
            self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self.image is None:
                raise ValueError(f"Could not read image at {image_path}")
        elif image_array is not None:
            self.image = image_array.copy()
        else:
            raise ValueError("Either image_path or image_array must be provided")
            
        # Convert to grayscale for some analyses
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        
        # Initialize results dictionary
        self.results = {}
    
    def analyze_pixel_distribution(self, save_dir: str = None) -> Dict:
        """
        Analyze the pixel value distribution of the image.
        
        Args:
            save_dir: Directory to save visualizations (if None, plots are displayed)
            
        Returns:
            Dictionary containing distribution statistics
        """
        # Calculate basic statistics
        stats = {
            'mean': np.mean(self.gray),
            'std': np.std(self.gray),
            'min': np.min(self.gray),
            'max': np.max(self.gray),
            'median': np.median(self.gray),
            'mode': float(scipy.stats.mode(self.gray.ravel(), keepdims=False).mode),
            'skewness': scipy.stats.skew(self.gray.ravel()),
            'kurtosis': scipy.stats.kurtosis(self.gray.ravel())
        }
        
        # Plot histogram
        plt.figure(figsize=(12, 6))
        plt.hist(self.gray.ravel(), bins=256, range=(0, 256), density=True, 
                color='blue', alpha=0.7, label='Pixel Intensity')
        plt.title('Pixel Intensity Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'pixel_distribution.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        self.results['pixel_distribution'] = stats
        return stats
    
    def analyze_correlations(self, save_dir: str = None) -> Dict:
        """
        Analyze pixel correlations in different directions.
        
        Args:
            save_dir: Directory to save visualizations (if None, plots are displayed)
            
        Returns:
            Dictionary containing correlation statistics
        """
        # Calculate correlations in different directions
        def calculate_correlations(img):
            h, w = img.shape
            
            # Horizontal correlation (shift right by 1 pixel)
            h_corr = np.corrcoef(img[:, :-1].ravel(), img[:, 1:].ravel())[0, 1]
            
            # Vertical correlation (shift down by 1 pixel)
            v_corr = np.corrcoef(img[:-1, :].ravel(), img[1:, :].ravel())[0, 1]
            
            # Diagonal correlation (shift right and down by 1 pixel)
            d1_corr = np.corrcoef(img[:-1, :-1].ravel(), img[1:, 1:].ravel())[0, 1]
            
            # Anti-diagonal correlation (shift left and down by 1 pixel)
            d2_corr = np.corrcoef(img[1:, :-1].ravel(), img[:-1, 1:].ravel())[0, 1]
            
            return {
                'horizontal': h_corr,
                'vertical': v_corr,
                'diagonal': d1_corr,
                'anti_diagonal': d2_corr
            }
        
        # Calculate correlations
        corr_results = calculate_correlations(self.gray)
        
        # Create visualization
        directions = list(corr_results.keys())
        values = list(corr_results.values())
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(directions, values, color='green', alpha=0.7)
        plt.title('Pixel Correlation in Different Directions')
        plt.ylabel('Correlation Coefficient')
        plt.ylim(-1, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'pixel_correlations.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        self.results['correlations'] = corr_results
        return corr_results
    
    def analyze_entropy(self, window_size: int = 7, save_dir: str = None) -> Dict:
        """
        Calculate local entropy of the image.
        
        Args:
            window_size: Size of the local window for entropy calculation
            save_dir: Directory to save visualizations (if None, plots are displayed)
            
        Returns:
            Dictionary containing entropy statistics
        """
        def local_entropy(image, window_size=7):
            """Calculate local entropy using a sliding window."""
            # Pad the image to handle borders
            pad = window_size // 2
            padded = np.pad(image, pad, mode='reflect')
            
            # Initialize output
            output = np.zeros_like(image, dtype=float)
            
            # Calculate local entropy
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Extract local window
                    window = padded[i:i+window_size, j:j+window_size]
                    # Calculate histogram
                    hist = np.histogram(window, bins=256, range=(0, 256))[0]
                    # Normalize to get probability distribution
                    prob = hist / (window_size * window_size)
                    # Calculate entropy (avoid log(0))
                    entropy = -np.sum(prob * np.log2(prob + 1e-10))
                    output[i, j] = entropy
            
            return output
        
        # Calculate local entropy
        local_entropy_map = local_entropy(self.gray, window_size)
        
        # Calculate global entropy
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        hist = hist / (self.gray.shape[0] * self.gray.shape[1])
        global_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        entropy_map = plt.imshow(local_entropy_map, cmap='viridis')
        plt.colorbar(entropy_map, label='Entropy')
        plt.title(f'Local Entropy (Window: {window_size}x{window_size})\nGlobal Entropy: {global_entropy:.2f} bits')
        plt.axis('off')
        
        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'entropy_analysis.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        entropy_results = {
            'global_entropy': float(global_entropy),
            'local_entropy_mean': float(np.mean(local_entropy_map)),
            'local_entropy_std': float(np.std(local_entropy_map)),
            'local_entropy_min': float(np.min(local_entropy_map)),
            'local_entropy_max': float(np.max(local_entropy_map))
        }
        
        self.results['entropy'] = entropy_results
        return entropy_results
    
    def analyze_differences(self, original_image_path: str = None, original_image: np.ndarray = None, 
                          save_dir: str = None) -> Dict:
        """
        Analyze differences between original and processed images.
        
        Args:
            original_image_path: Path to the original image
            original_image: Numpy array of the original image
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing difference statistics
        """
        if original_image is None and original_image_path:
            original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise ValueError(f"Could not read original image at {original_image_path}")
        elif original_image is None:
            raise ValueError("Either original_image_path or original_image must be provided")
        
        # Resize images to match dimensions if needed
        if original_image.shape != self.gray.shape:
            original_image = cv2.resize(original_image, (self.gray.shape[1], self.gray.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(original_image, self.gray)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_score = ssim(original_image, self.gray, 
                         data_range=255,  # For uint8 images
                         gaussian_weights=True,
                         sigma=1.5,
                         use_sample_covariance=False)
        
        # Calculate MSE and PSNR
        mse = np.mean((original_image.astype(float) - self.gray.astype(float)) ** 2)
        if mse == 0:  # Images are identical
            psnr = 100
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed image
        plt.subplot(1, 3, 2)
        plt.imshow(self.gray, cmap='gray')
        plt.title('Processed Image')
        plt.axis('off')
        
        # Difference
        plt.subplot(1, 3, 3)
        diff_plot = plt.imshow(diff, cmap='hot')
        plt.colorbar(diff_plot, label='Difference')
        plt.title(f'Difference\nSSIM: {ssim_score:.3f}, PSNR: {psnr:.1f} dB')
        plt.axis('off')
        
        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'difference_analysis.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        diff_results = {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim_score),
            'mean_diff': float(np.mean(diff)),
            'max_diff': float(np.max(diff)),
            'diff_std': float(np.std(diff))
        }
        
        self.results['differences'] = diff_results
        return diff_results
    
    def run_all_analyses(self, original_image_path: str = None, original_image: np.ndarray = None, 
                        save_dir: str = 'analysis_results') -> Dict:
        """
        Run all available analyses on the image.
        
        Args:
            original_image_path: Path to the original image (for difference analysis)
            original_image: Numpy array of the original image (alternative to path)
            save_dir: Directory to save all visualizations
            
        Returns:
            Dictionary containing all analysis results
        """
        # Create output directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Run all analyses
        self.analyze_pixel_distribution(save_dir=save_dir)
        self.analyze_correlations(save_dir=save_dir)
        self.analyze_entropy(save_dir=save_dir)
        
        # Only run difference analysis if original image is provided
        if original_image_path is not None or original_image is not None:
            try:
                self.analyze_differences(
                    original_image_path=original_image_path,
                    original_image=original_image,
                    save_dir=save_dir
                )
            except Exception as e:
                print(f"Error in difference analysis: {e}")
        
        # Save results to a text file
        if save_dir:
            with open(os.path.join(save_dir, 'analysis_summary.txt'), 'w') as f:
                import json
                f.write("Image Analysis Results\n")
                f.write("="*50 + "\n\n")
                
                # Pixel Distribution
                f.write("1. Pixel Distribution Analysis\n")
                f.write("-"*40 + "\n")
                dist_stats = self.results.get('pixel_distribution', {})
                for key, value in dist_stats.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                # Correlations
                f.write("2. Pixel Correlation Analysis\n")
                f.write("-"*40 + "\n")
                corr_stats = self.results.get('correlations', {})
                for key, value in corr_stats.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
                
                # Entropy
                f.write("3. Entropy Analysis\n")
                f.write("-"*40 + "\n")
                entropy_stats = self.results.get('entropy', {})
                for key, value in entropy_stats.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
                
                # Differences (if available)
                if 'differences' in self.results:
                    f.write("4. Difference Analysis\n")
                    f.write("-"*40 + "\n")
                    diff_stats = self.results['differences']
                    for key, value in diff_stats.items():
                        f.write(f"{key.upper()}: {value:.4f}\n")
                
                f.write("\nAnalysis completed successfully!")
        
        return self.results


def analyze_image(image_path: str, original_image_path: str = None, save_dir: str = 'analysis_results') -> Dict:
    """
    Convenience function to analyze a single image.
    
    Args:
        image_path: Path to the image to analyze
        original_image_path: Optional path to the original image (for difference analysis)
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing all analysis results
    """
    analyzer = ImageAnalyzer(image_path=image_path)
    return analyzer.run_all_analyses(
        original_image_path=original_image_path,
        save_dir=save_dir
    )


def compare_images(image1_path: str, image2_path: str, save_dir: str = 'comparison_results') -> Dict:
    """
    Compare two images and generate analysis.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        save_dir: Directory to save comparison results
        
    Returns:
        Dictionary containing comparison results
    """
    # Analyze first image
    analyzer1 = ImageAnalyzer(image_path=image1_path)
    results1 = analyzer1.run_all_analyses(save_dir=os.path.join(save_dir, 'image1_analysis'))
    
    # Analyze second image
    analyzer2 = ImageAnalyzer(image_path=image2_path)
    results2 = analyzer2.run_all_analyses(save_dir=os.path.join(save_dir, 'image2_analysis'))
    
    # Compare the two images
    analyzer2.analyze_differences(
        original_image=cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE),
        save_dir=os.path.join(save_dir, 'difference_analysis')
    )
    
    # Save comparison summary
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("Image Comparison Summary\n")
        f.write("="*50 + "\n\n")
        
        # Global entropy comparison
        ent1 = results1.get('entropy', {}).get('global_entropy', 0)
        ent2 = results2.get('entropy', {}).get('global_entropy', 0)
        f.write(f"Global Entropy:\n  Image 1: {ent1:.4f}\n  Image 2: {ent2:.4f}\n  Difference: {abs(ent1 - ent2):.4f}\n\n")
        
        # Pixel distribution comparison
        f.write("Pixel Distribution:\n")
        stats1 = results1.get('pixel_distribution', {})
        stats2 = results2.get('pixel_distribution', {})
        for key in ['mean', 'std', 'skewness', 'kurtosis']:
            if key in stats1 and key in stats2:
                f.write(f"  {key.title()}: {stats1[key]:.4f} vs {stats2[key]:.4f} (Δ: {abs(stats1[key] - stats2[key]):.4f})\n")
        
        # Correlation comparison
        f.write("\nCorrelation Analysis:\n")
        corr1 = results1.get('correlations', {})
        corr2 = results2.get('correlations', {})
        for key in ['horizontal', 'vertical', 'diagonal', 'anti_diagonal']:
            if key in corr1 and key in corr2:
                f.write(f"  {key.title()}: {corr1[key]:.4f} vs {corr2[key]:.4f} (Δ: {abs(corr1[key] - corr2[key]):.4f})\n")
        
        # Difference metrics
        diff = analyzer2.results.get('differences', {})
        if diff:
            f.write("\nDifference Metrics:\n")
            for key, value in diff.items():
                f.write(f"  {key.upper()}: {value:.4f}\n")
    
    return {
        'image1_results': results1,
        'image2_results': results2,
        'difference_metrics': analyzer2.results.get('differences', {})
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze image properties for AI resistance evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single image')
    analyze_parser.add_argument('image_path', help='Path to the image to analyze')
    analyze_parser.add_argument('--original', help='Path to the original image (for difference analysis)')
    analyze_parser.add_argument('--output', default='analysis_results', 
                              help='Directory to save analysis results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two images')
    compare_parser.add_argument('image1', help='Path to the first image')
    compare_parser.add_argument('image2', help='Path to the second image')
    compare_parser.add_argument('--output', default='comparison_results',
                              help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        print(f"Analyzing image: {args.image_path}")
        if args.original:
            print(f"Comparing with original image: {args.original}")
        print(f"Saving results to: {args.output}")
        
        analyzer = ImageAnalyzer(image_path=args.image_path)
        results = analyzer.run_all_analyses(
            original_image_path=args.original,
            save_dir=args.output
        )
        print("Analysis complete!")
        
    elif args.command == 'compare':
        print(f"Comparing images:\n  1. {args.image1}\n  2. {args.image2}")
        print(f"Saving results to: {args.output}")
        
        results = compare_images(
            image1_path=args.image1,
            image2_path=args.image2,
            save_dir=args.output
        )
        print("Comparison complete!")
        
    else:
        parser.print_help()
