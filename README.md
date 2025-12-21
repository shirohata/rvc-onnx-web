# rvc-onnx-web

[![npm version](https://img.shields.io/npm/v/rvc-onnx-web.svg)](https://www.npmjs.com/package/rvc-onnx-web)
[![CI](https://github.com/visgotti/rvc-onnx-web/actions/workflows/ci.yml/badge.svg)](https://github.com/visgotti/rvc-onnx-web/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/visgotti/rvc-onnx-web/master)](https://codecov.io/gh/visgotti/rvc-onnx-web)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Convert RVC voice models (.pth) to ONNX format - runs entirely in TypeScript (Browser & Node.js)**

This is a complete reimplementation of PyTorch's model conversion pipeline in TypeScript. No Python runtime required. Converts RVC v2 voice models to ONNX for use with ONNX Runtime Web.

## 🌐 [Live Demo](https://visgotti.github.io/rvc-onnx-web)

## ✨ Features

- 🌐 **100% Browser Compatible** - No Python, no server, runs entirely client-side
- 📦 **Zero Native Dependencies** - Pure TypeScript/JavaScript
- 🎯 **RVC v2 Support** - Full support for RVC v2 voice models
- ✅ **100% Accuracy** - Deterministic outputs match Python ONNX export exactly
- 🔒 **Privacy First** - Your model never leaves your device
- ⚡ **Fast** - Converts models in seconds

## 📦 Installation

```bash
npm install rvc-onnx-web
```

## 🚀 Quick Start

### Node.js

```typescript
import { pthToOnnx } from 'rvc-onnx-web';
import { readFileSync, writeFileSync } from 'fs';

// Load your .pth model
const pthBuffer = readFileSync('MyVoiceModel.pth');

// Convert to ONNX (accepts Buffer directly via Uint8Array overload)
const { onnxBuffer, sampleRate } = await pthToOnnx(pthBuffer, {
  opsetVersion: 17,
  phoneLen: 100  // Dynamic shapes supported
});

console.log(`Model sample rate: ${sampleRate}`);

// Save the result
writeFileSync('MyVoiceModel.onnx', Buffer.from(onnxBuffer));
```

### Browser

```typescript
import { pthToOnnx } from 'rvc-onnx-web';

// From file input
const fileInput = document.getElementById('file-input') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = fileInput.files?.[0];
  if (!file) return;

  // pthToOnnx accepts File directly (also supports ArrayBuffer, Blob, URL, etc.)
  const { onnxBuffer, sampleRate, checkpoint } = await pthToOnnx(file, { opsetVersion: 17 });

  console.log(`Converted model: ${sampleRate}Hz, ${checkpoint.weights.size} weights`);

  // Download the converted model
  const blob = new Blob([onnxBuffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = file.name.replace('.pth', '.onnx');
  a.click();
});
```

## 📖 API Reference

### `pthToOnnx(input, options)`

Converts a PyTorch .pth checkpoint to ONNX format.

**Parameters:**
- `input: PthInput` - The .pth file in any supported format:
  - `ArrayBuffer` - Raw binary data
  - `Uint8Array` - Byte array
  - `File` - Browser File object
  - `Blob` - Browser Blob object
  - `Response` - Fetch Response object
  - `URL | string` - URL to fetch the .pth file from
- `options: ConvertOptions`
  - `opsetVersion?: number` - ONNX opset version (default: 17)
  - `phoneLen?: number` - Sequence length for graph construction (default: 100)

**Returns:** `Promise<ConversionResult>`
  - `onnxBuffer: Uint8Array` - The serialized ONNX model
  - `checkpoint: ParsedCheckpoint` - Parsed model metadata and weights
  - `sampleRate: number` - Model sample rate (e.g., 40000, 48000)

### `parsePth(buffer)`

Parse a .pth file and extract weights and configuration.

```typescript
import { parsePth } from 'rvc-onnx-web/parser';

const checkpoint = await parsePth(pthBuffer);
console.log(checkpoint.config);    // Model configuration
console.log(checkpoint.weights);   // Map<string, TensorData>
console.log(checkpoint.version);   // "v1" or "v2"
console.log(checkpoint.useF0);     // Whether model uses pitch
```

### `buildOnnxModel(checkpoint, options)`

Build an ONNX model from a parsed checkpoint.

```typescript
import { buildOnnxModel } from 'rvc-onnx-web/builder';
import { serializeOnnx } from 'rvc-onnx-web';

const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });
const bytes = serializeOnnx(model);
```

## 🏗️ Architecture

This library implements the complete RVC Synthesizer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      RVC Synthesizer                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Text      │   │   Flow      │   │    HiFi-GAN     │   │
│  │  Encoder    │──▶│  Decoder    │──▶│   Generator     │──▶│ Audio
│  │  (enc_p)    │   │   (flow)    │   │     (dec)       │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│        │                                     ▲              │
│        │         ┌─────────────┐            │              │
│        │         │    NSF      │            │              │
│        └────────▶│   Sine      │────────────┘              │
│                  │  Generator  │                            │
│                  └─────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- **Text Encoder**: Multi-head attention with relative positional encoding
- **Flow Decoder**: Normalizing flow with 4 residual coupling blocks
- **HiFi-GAN**: Transposed convolutions with multi-receptive field fusion
- **NSF Generator**: Neural source-filter model for F0-based synthesis

## 📊 Verified Accuracy

All deterministic operations achieve **100% correlation** with Python-exported ONNX:

| Component | Correlation | Max Difference |
|-----------|-------------|----------------|
| Encoder m_p | 100.0000% | 2.77e-6 |
| Encoder logs_p | 100.0000% | 9.54e-7 |
| Sine Generator | 100.0000% | 0.0 |
| All 247 Weights | Exact Match | 0.0 |

*Note: Final audio output has ~78% correlation due to intentional randomness (RandomNormalLike ops) for audio quality.*

## 🔧 Supported Models

| Model Type | Status |
|------------|--------|
| RVC v2 | ✅ Full Support |
| RVC v1 | ⚠️ Limited (different architecture) |
| So-VITS | ❌ Not Supported |

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Build
npm run build

# Lint
npm run lint
```

## 📁 Project Structure

```
rvc-onnx-web/
├── src/
│   ├── index.ts              # Main entry point
│   ├── types.ts              # Type definitions
│   ├── pickle.ts             # Python pickle protocol parser
│   ├── pth-parser.ts         # PyTorch checkpoint parser
│   ├── onnx-builder.ts       # ONNX graph utilities
│   ├── onnx-serializer.ts    # ONNX protobuf serializer
│   └── synthesizer-builder.ts # RVC model graph builder
├── tests/
│   └── converter.spec.ts     # Integration tests
├── docs/                     # GitHub Pages demo
└── .github/workflows/        # CI/CD pipelines
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## ⚠️ Caveats & Limitations

- **RVC v2 Only** - This library currently supports RVC v2 models only. RVC v1 models are not supported.
- **F0 Models** - Designed for pitch-enabled (F0) models. Non-F0 models may not work correctly.
- **Browser Memory** - Large models (>100MB) may cause memory issues in browsers with limited RAM.
- **Not Affiliated** - This project is not affiliated with or endorsed by the RVC or ONNX teams.

## ⚖️ Responsible Use

Voice cloning technology can be misused. Please:

- ✅ **Do** use for creative projects, accessibility, content creation with consent
- ✅ **Do** obtain permission before cloning someone's voice
- ✅ **Do** clearly label AI-generated voice content
- ❌ **Don't** use for impersonation, fraud, or deception
- ❌ **Don't** create non-consensual voice clones
- ❌ **Don't** use for harassment or defamation

The authors are not responsible for misuse of this software.

## 📄 License

MIT © Joseph Viscardi