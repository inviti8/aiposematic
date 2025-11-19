# AIPosematic: Overt Adversarial Protection for Digital Art

AIPosematic is a Python package inspired by nature's defense mechanisms, designed to help artists and creators protect their digital artwork in an overt and visually distinctive way. Unlike traditional digital watermarking or covert adversarial examples, AIPosematic applies visible, intentional transformations that signal the work is protected while simultaneously disrupting AI training processes.

## How It Works

AIPosematic employs a multi-layered approach to protect digital images:

1. **Visual Signature**
   - Applies a unique, visible pattern to the image that serves as a warning to AI systems
   - The pattern is designed to be aesthetically integrated while remaining clearly artificial
   - Functions as a "digital aposematism" - a warning signal in the digital ecosystem

2. **Cryptographic Implementation**
   - **Secure Key Generation**: Utilizes cryptographically secure 128-bit keys (2^128 possible combinations) for encryption
   - **Dual Key System**: Combines a unique image key with a 128-bit cipher key for each protected image
   - **Operation Chaining**: Applies a sequence of reversible operations (XOR, addition, subtraction, bit rotations) defined by a customizable operation string
   - **Steganographic Embedding**: The key image is embedded using a custom algorithm that preserves visual quality while ensuring recoverability
   - **Defense-in-Depth**: Implements multiple layers of protection including spatial obfuscation and pixel-level transformations

3. **Dual Protection**
   - **Human-Visible**: The protection is intentionally visible to establish clear provenance
   - **AI-Disruptive**: The transformations are designed to confuse and degrade the performance of AI models
   - **Reversible**: Original image can be recovered with the proper key and transformation sequence

## Why It Works: Technical Advantages Over Other Approaches

### The Problem with Traditional Adversarial Poisoning

Most AI protection tools use subtle, invisible perturbations to poison training data. These approaches have several weaknesses:

1. **Pattern Recognition Vulnerability**: AI models are trained to recognize and potentially learn to ignore small perturbations
2. **Dilution Effect**: A few poisoned samples in a large dataset have minimal impact on model training
3. **Ethical Concerns**: Invisible modifications can be seen as deceptive and may have unintended consequences

### How AIPosematic is Different

1. **Overt, Not Covert**
   - Unlike adversarial examples that rely on subtle perturbations, AIPosematic's protection is intentionally visible
   - This establishes clear intent and provenance, similar to how aposematic coloring in nature warns predators
   - The visible nature makes it immediately apparent that the image is protected

2. **Unique Per-Image Protection**
   - Each image receives a unique scrambling pattern generated through high-entropy processes:
     - **BUTTERFLY Mode**: Generates mathematically-derived parametric curves with random positions, scales, and rotations, creating visually complex patterns that appear as high-entropy noise to AI systems
     - **QR Mode**: Produces multiple overlapping QR codes with random data, sizes, and orientations, resulting in a dense field of machine-readable glyphs that appear as visual noise
   - Both methods incorporate:
     - Cryptographically secure 128-bit cipher keys (2^128 possible combinations)
     - Randomly distributed visual elements with controlled spatial frequencies
     - Non-linear transformations that disrupt feature extraction
   - The combination of these elements ensures each image's protection is unique and resistant to pattern recognition, making it impossible to train models to remove or bypass the protection

3. **Multi-Dimensional Disruption**
   - Our statistical analysis shows that AIPosematic effectively:
     - Reduces spatial correlations from ~0.95 to ~0.07 (92% reduction)
     - Maintains high entropy (7.73 → 7.72), preserving randomness
     - Introduces significant visual differences (MSE > 6900, SSIM < 0.05)
   - These metrics demonstrate strong disruption of features that AI models rely on

4. **Ethical and Transparent**
   - No hidden modifications or "poisoning" that could have unintended consequences
   - Clear visual indication of protection status
   - Reversible with proper authorization, respecting fair use cases

### Comparison with Nightshade and Similar Tools

| Feature | AIPosematic | Nightshade/Others |
|---------|-------------|-------------------|
| Visibility | Overt and visible | Invisible modifications |
| Protection Type | Per-image unique pattern | Universal perturbation |
| Impact on AI Training | Disrupts feature extraction | Attempts to poison specific concepts |
| Ethical Transparency | High (visible protection) | Low (hidden modifications) |
| Reversibility | Fully reversible with key | Typically irreversible |
| Statistical Impact | Preserves entropy, disrupts correlations | May reduce image quality |
| Defense Against | Both training and inference | Primarily training |

### Technical Superiority

1. **Resistant to Mitigation**
   - Unlike pattern-based protections, AIPosematic's unique per-image scrambling cannot be "learned around" by AI models
   - The combination of spatial and value-space transformations creates a moving target for any attempted removal

2. **Preserves Image Quality**
   - While disrupting AI training, the protected images remain visually clear and usable for human viewers
   - The protection is integrated as an artistic element rather than degradation

3. **Future-Proof Design**
   - The cryptographic foundation ensures that even as AI models improve, the protection remains effective
   - The system is designed to be adaptable, with the ability to update transformation techniques as needed

By combining cryptographic security with visual signaling, AIPosematic provides a robust, ethical, and effective solution for artists and creators to protect their work in the age of generative AI.

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
