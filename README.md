# AIPosematic: Overt Adversarial Protection for Digital Art

AIPosematic is a Python package inspired by nature's defense mechanisms, designed to help artists and creators protect their digital artwork in an overt and visually distinctive way. Unlike traditional digital watermarking or covert adversarial examples, AIPosematic applies visible, intentional transformations that signal the work is protected while simultaneously disrupting AI training processes.

## How It Works

AIPosematic employs a multi-layered approach to protect digital images:

1. **Visual Signature**
   - Applies a unique, visible pattern to the image that serves as a warning to AI systems
   - The pattern is designed to be aesthetically integrated while remaining clearly artificial
   - Functions as a "digital aposematism" - a warning signal in the digital ecosystem

2. **Technical Implementation**
   - **Pixel Shuffling**: Rearranges pixels using a deterministic but non-obvious pattern
   - **Key-Based Recovery**: Uses a separate key image to reverse the transformation
   - **High-Frequency Noise**: Adds subtle noise patterns that disrupt feature extraction
   - **Edge Manipulation**: Modifies edge regions to confuse edge detection algorithms
   - **QR Code Integration**: Embeds recovery information in visually integrated QR codes

3. **Dual Protection**
   - **Human-Visible**: The protection is intentionally visible to establish clear provenance
   - **AI-Disruptive**: The transformations are designed to confuse and degrade the performance of AI models
   - **Reversible**: Original image can be recovered with the proper key and transformation sequence

## Installation

```bash
pip install aiposematic
```

## Basic Usage

```python
from aiposematic import new_aposematic_img, recover_aposematic_img

# Protect an image
result = new_aposematic_img(
    "original.png",
    op_string='-^+',  # Transformation operations
    scramble_mode='QR'  # Key generation mode
)

# The protected image and cipher key are saved
print(f"Protected image: {result['img_path']}")
print(f"Cipher key: {result['cipher_key']}")

# Recover the original image
recovered_path = recover_aposematic_img(
    result['img_path'],
    cipher_key=result['cipher_key']
)
print(f"Recovered image: {recovered_path}")
```

## Key Features

- **Multiple Scrambling Modes**:
  - `BUTTERFLY`: Creates a butterfly pattern key meant to confuse AI models
  - `QR`: Generates a pattern of QR codes meant to confuse AI models

- **Customizable Transformations**:
  - Chain multiple operations (+, -, ^, etc.) for custom protection schemes
  - Adjust intensity and visibility of protection patterns

- **High-Quality Output**:
  - Preserves image quality for human viewers
  - Maintains compatibility with standard image formats

## Use Cases

- **Digital Art Protection**: Clearly mark and protect digital artwork
- **Dataset Poisoning**: Create "do not train" markers for AI datasets
- **Provenance Tracking**: Embed recoverable ownership information
- **Ethical AI Development**: Create clear visual indicators of usage restrictions

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## Acknowledgements

Inspired by natural aposematism and the need for better digital rights management in the age of generative AI.
