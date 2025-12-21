# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-20

### Added
- Initial release
- Full Python pickle protocol parser (protocols 0-5)
- PyTorch .pth checkpoint parser with weight normalization support
- Complete RVC v2 Synthesizer architecture:
  - Text Encoder with multi-head attention and relative positional encoding
  - Normalizing Flow decoder with ResidualCouplingBlocks
  - HiFi-GAN vocoder with multi-receptive field fusion
  - NSF sine generator for F0-based synthesis
- ONNX protobuf serializer (no external dependencies)
- Browser-based demo application
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- GitHub Pages deployment

### Verified
- 100% correlation on deterministic operations vs Python ONNX export
- All 247 weights match exactly (0.0 difference)
- Dynamic shape support tested with multiple sequence lengths
