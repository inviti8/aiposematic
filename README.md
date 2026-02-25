# AIPosematic v1.0: Overt Adversarial Protection for Digital Art

AIPosematic is a Python package inspired by nature's defense mechanisms, designed to help artists and creators protect their digital artwork in an overt and visually distinctive way. Unlike traditional digital watermarking or covert adversarial examples, AIPosematic applies visible, intentional transformations that signal the work is protected while simultaneously disrupting AI training processes.

## How It Works

AIPosematic employs a multi-layered approach to protect digital images:

1. **Visual Signature**
   - Applies a unique, visible pattern to the image that serves as a warning to AI systems
   - The pattern is designed to be aesthetically integrated while remaining clearly artificial
   - Functions as a "digital aposematism" - a warning signal in the digital ecosystem

2. **Cryptographic Implementation (v1.0)**
   - **Secure Key Generation**: Generates a unique butterfly or QR code key image per image, used as the cryptographic key for pixel operations while preserving the aposematic visual quality
   - **128-bit Cipher Key**: Each image gets a unique cipher key (`secrets.token_hex(16)`) used for S-box derivation and key image shuffling
   - **Key-Dependent Operations**: All 9 operations are fully key-dependent — rotations use 1-7 bits determined by the key pixel, the S-box is derived per-image from the cipher key, and cross-channel operations use the full key byte
   - **Per-Image S-box**: The substitution table is derived deterministically from the cipher key via seeded Fisher-Yates shuffle, eliminating the v0.4 session-dependent global S-box
   - **Steganographic Embedding**: The key image is shuffled using the cipher key and embedded in the scrambled image for recovery
   - **Defense-in-Depth**: Multiple layers of protection including AI deterrent features, spatial obfuscation, and pixel-level key-dependent transformations

3. **Dual Protection**
   - **Human-Visible**: The protection is intentionally visible to establish clear provenance
   - **AI-Disruptive**: The transformations are designed to confuse and degrade the performance of AI models
   - **Reversible**: Original image can be recovered with the proper cipher key and operation string

## v1.0 Security Improvements Over v0.4

| Issue | v0.4 | v1.0 |
|:------|:-----|:-----|
| **Keyless ops** (`>`, `<`, `p`, `P`) | Ignored key entirely — immediately reversible without secret | All key-dependent: rotation by 1-7 bits from key, XOR-whitened S-box lookup |
| **Weak ops** (`a`, `A`) | 4-bit R-channel only (16 brute-force attempts) | Cross-channel sum of all 3 key bytes (full 8-bit range) |
| **Global S-box** | `np.random.permutation(256)` at import — session-random, shared across all images | Per-image S-box derived from cipher key via `_derive_sbox()` |
| **Mixed op_string security** | Adding keyless ops diluted security — `"-^+p>a"` had only 33% encrypted pixels | All 9 ops are key-dependent — 100% of pixels are encrypted for any op_string |

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
     - Per-image S-box derived from cipher key
     - All key-dependent operations ensuring every pixel is protected
   - The combination of these elements ensures each image's protection is unique and resistant to pattern recognition

3. **Multi-Dimensional Disruption**
   - Our statistical analysis shows that AIPosematic effectively:
     - Reduces spatial correlations from ~0.95 to ~0.07 (92% reduction)
     - Maintains high entropy (7.73 -> 7.72), preserving randomness
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
| Protection Type | Per-image unique pattern + key | Universal perturbation |
| Impact on AI Training | Disrupts feature extraction | Attempts to poison specific concepts |
| Ethical Transparency | High (visible protection) | Low (hidden modifications) |
| Reversibility | Fully reversible with key | Typically irreversible |
| Statistical Impact | Preserves entropy, disrupts correlations | May reduce image quality |
| Defense Against | Both training and inference | Primarily training |

## Installation

```bash
pip install aiposematic
```

## Basic Usage

```python
from aiposematic import new_aposematic_img, recover_aposematic_img, SCRAMBLE_MODE

# Protect an image
result = new_aposematic_img(
    "original.png",
    op_string='-^+',                       # Transformation operations
    scramble_mode=SCRAMBLE_MODE.BUTTERFLY   # Key generation mode
)

# The protected image and cipher key are returned
print(f"Protected image: {result['img_path']}")
print(f"Cipher key: {result['cipher_key']}")

# Recover the original image
recovered_path = recover_aposematic_img(
    result['img_path'],
    cipher_key=result['cipher_key'],
    op_string='-^+'
)
print(f"Recovered image: {recovered_path}")
```

### Low-Level API

```python
from aiposematic import scramble, recover

# Scramble (generates butterfly key image + cipher key)
result = scramble("original.png", op_string="-^+p>a")
print(f"Cipher key: {result['cipher_key']}")
print(f"Key image: {result['key_path']}")

# Recover with the same key image, cipher key, and op_string
recovered = recover(
    result['scrambled_path'],
    key_img_path=result['key_path'],
    cipher_key=result['cipher_key'],
    op_string="-^+p>a"
)
```

## Operations

All 9 operations are key-dependent in v1.0:

| Op | Name | Description |
|:---|:-----|:------------|
| `+` | Add | `(pixel + key) mod 256` per channel |
| `-` | Subtract | `(pixel - key) mod 256` per channel |
| `^` | XOR | `pixel XOR key` per channel |
| `>` | Rotate Right | Rotate right by `(key % 7) + 1` bits per channel |
| `<` | Rotate Left | Rotate left by `(key % 7) + 1` bits per channel |
| `p` | S-box | `SBOX[pixel XOR key]` — per-image S-box derived from cipher key |
| `P` | Inverse S-box | `SBOX_INV[pixel] XOR key` |
| `a` | Channel Add | `(pixel + sum(key_r, key_g, key_b)) mod 256` — cross-channel |
| `A` | Channel Sub | `(pixel - sum(key_r, key_g, key_b)) mod 256` — cross-channel |

Operations are applied via partitioning: each pixel receives one operation from the op_string cycle (`pixel_i` gets `op_string[i % len(op_string)]`). Since all operations are key-dependent, every pixel is protected regardless of which operation it receives.

## Key Features

- **Multiple Key Generation Modes**:
  - `SCRAMBLE_MODE.BUTTERFLY`: Creates a butterfly pattern key image
  - `SCRAMBLE_MODE.QR`: Generates a QR code pattern key image

- **Hardened Cryptographic Operations**:
  - All 9 operations are key-dependent (v0.4 had 4 keyless operations)
  - Per-image S-box derived from cipher key (v0.4 used a global session-random S-box)
  - Cross-channel operations use full 8-bit key (v0.4 used only 4-bit R-channel)

- **Customizable Transformations**:
  - Chain any combination of 9 operations (`+`, `-`, `^`, `>`, `<`, `p`, `P`, `a`, `A`)
  - More operations in the op_string now means more security, not less

- **High-Quality Output**:
  - Lossless PNG output preserves pixel-perfect recoverability
  - AI deterrent features (high-frequency noise, adversarial edge patterns) applied before scrambling

## Security Properties

- **No keyless operations**: Every operation depends on the key image pixels (v0.4 had 4 keyless ops that could be reversed without any secret)
- **Per-image S-box**: Deterministic from cipher key, unique per image, eliminates session-dependency bug
- **Full op_string security**: Any op_string provides 100% encrypted pixels (v0.4: `"-^+p>a"` had only 33% encrypted)
- **Operation-aware resistance**: Knowing the op_string without the key gives no advantage (Kerckhoffs' principle)

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
