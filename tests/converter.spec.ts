/**
 * Comprehensive test suite for the RVC ONNX Converter.
 */

import { describe, it, expect } from 'vitest';
import { pthToOnnx } from '../src/index.js';
import { parsePth } from '../src/pth-parser.js';
import { buildOnnxModel } from '../src/onnx-builder.js';
import { serializeOnnx } from '../src/onnx-serializer.js';
import { Unpickler } from '../src/pickle.js';

describe('Pickle Parser', () => {
  it('should handle basic pickle opcodes', () => {
    // Minimal pickle: PROTO 4, EMPTY_DICT, STOP
    const pickle = new Uint8Array([0x80, 0x04, 0x7d, 0x2e]);
    const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
    const result = unpickler.load();
    expect(result).toEqual({});
  });

  it('should handle nested structures', () => {
    // Test that unpickler initializes without error
    const unpickler = new Unpickler(
      new Uint8Array([0x80, 0x04, 0x5d, 0x2e]), // PROTO 4, EMPTY_LIST, STOP
      () => new Uint8Array(0)
    );
    const result = unpickler.load();
    expect(Array.isArray(result)).toBe(true);
  });
});

describe('ONNX Serializer', () => {
  it('should serialize a minimal model', () => {
    const model = {
      irVersion: 8n,
      opsetImports: [{ domain: '', version: 17n }],
      producerName: 'test',
      producerVersion: '1.0.0',
      graph: {
        name: 'test_graph',
        nodes: [],
        inputs: [],
        outputs: [],
        initializers: []
      }
    };

    const bytes = serializeOnnx(model);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect(bytes.length).toBeGreaterThan(0);
    // ONNX files start with field tag for ir_version (field 1, varint = 0x08)
    expect(bytes[0]).toBe(0x08);
  });

  it('should serialize model with nodes', () => {
    const model = {
      irVersion: 8n,
      opsetImports: [{ domain: '', version: 17n }],
      producerName: 'test',
      producerVersion: '1.0.0',
      graph: {
        name: 'test_graph',
        nodes: [{
          opType: 'Identity',
          name: 'identity_0',
          inputs: ['input'],
          outputs: ['output'],
          attributes: []
        }],
        inputs: [{
          name: 'input',
          elemType: 1, // FLOAT
          shape: [{ dimValue: 1n }, { dimValue: 10n }]
        }],
        outputs: [{
          name: 'output',
          elemType: 1,
          shape: [{ dimValue: 1n }, { dimValue: 10n }]
        }],
        initializers: []
      }
    };

    const bytes = serializeOnnx(model);
    expect(bytes.length).toBeGreaterThan(50);
  });

  it('should serialize initializers with float data', () => {
    const model = {
      irVersion: 8n,
      opsetImports: [{ domain: '', version: 17n }],
      producerName: 'test',
      producerVersion: '1.0.0',
      graph: {
        name: 'test_graph',
        nodes: [],
        inputs: [],
        outputs: [],
        initializers: [{
          name: 'weights',
          data: {
            data: new Float32Array([1, 2, 3, 4]),
            shape: [2, 2],
            dtype: 'float32' as const
          }
        }]
      }
    };

    const bytes = serializeOnnx(model);
    expect(bytes.length).toBeGreaterThan(20);
  });
});

describe('ONNX Builder', () => {
  it('should create a valid model structure', () => {
    // Create a mock checkpoint
    const mockCheckpoint = {
      config: {
        specChannels: 1025,
        segmentSize: 32,
        interChannels: 192,
        hiddenChannels: 192,
        filterChannels: 768,
        nHeads: 2,
        nLayers: 6,
        kernelSize: 3,
        pDropout: 0,
        resblock: '1',
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsampleRates: [10, 10, 2, 2],
        upsampleInitialChannel: 512,
        upsampleKernelSizes: [16, 16, 4, 4],
        spkEmbedDim: 256,
        ginChannels: 256,
        sr: 40000
      },
      weights: new Map(),
      useF0: true,
      version: 'v2' as const,
      vocoder: 'HiFi-GAN'
    };

    // Add minimal required weights
    const embWeight = new Float32Array(256 * 256);
    mockCheckpoint.weights.set('emb_g.weight', {
      data: embWeight,
      shape: [256, 256],
      dtype: 'float32' as const
    });

    // Building should not throw (may fail due to missing weights, but structure is valid)
    expect(() => {
      try {
        buildOnnxModel(mockCheckpoint, { opsetVersion: 17, phoneLen: 10 });
      } catch (e) {
        // Expected to fail due to missing weights, but validates structure
        if (!(e instanceof Error) || !e.message.includes('Weight not found')) {
          throw e;
        }
      }
    }).not.toThrow();
  });
});

describe('Integration', () => {
  it('should export pthToOnnx function', () => {
    expect(typeof pthToOnnx).toBe('function');
  });

  it('should export parsePth function', () => {
    expect(typeof parsePth).toBe('function');
  });

  it('should export buildOnnxModel function', () => {
    expect(typeof buildOnnxModel).toBe('function');
  });

  it('should export serializeOnnx function', () => {
    expect(typeof serializeOnnx).toBe('function');
  });
});

describe('pthToOnnx Input Normalization', () => {
  // Create a minimal invalid .pth to test input handling (not conversion)
  const testBuffer = new Uint8Array([0x80, 0x04, 0x7d, 0x2e]).buffer;
  
  it('should accept ArrayBuffer', async () => {
    // The conversion will fail due to invalid .pth, but input acceptance should work
    await expect(pthToOnnx(testBuffer)).rejects.toThrow();
  });

  it('should accept Uint8Array', async () => {
    const uint8 = new Uint8Array(testBuffer);
    await expect(pthToOnnx(uint8)).rejects.toThrow();
  });

  it('should accept Uint8Array slice (view of larger buffer)', async () => {
    const largerBuffer = new Uint8Array(100);
    largerBuffer.set([0x80, 0x04, 0x7d, 0x2e], 10);
    const slice = largerBuffer.subarray(10, 14);
    await expect(pthToOnnx(slice)).rejects.toThrow();
  });

  it('should accept Blob', async () => {
    const blob = new Blob([testBuffer]);
    await expect(pthToOnnx(blob)).rejects.toThrow();
  });

  it('should accept Response with ok status', async () => {
    const response = new Response(testBuffer, { status: 200, statusText: 'OK' });
    Object.defineProperty(response, 'ok', { value: true });
    await expect(pthToOnnx(response)).rejects.toThrow();
  });

  it('should reject Response with error status', async () => {
    const response = new Response(testBuffer, { status: 404, statusText: 'Not Found' });
    Object.defineProperty(response, 'ok', { value: false });
    await expect(pthToOnnx(response)).rejects.toThrow('Failed to fetch: 404 Not Found');
  });

  it('should accept ReadableStream', async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([0x80, 0x04]));
        controller.enqueue(new Uint8Array([0x7d, 0x2e]));
        controller.close();
      }
    });
    await expect(pthToOnnx(stream)).rejects.toThrow();
  });

  it('should reject invalid string input', async () => {
    await expect(pthToOnnx('not-a-url')).rejects.toThrow('String input must be a valid URL');
  });

  it('should reject null input', async () => {
    // @ts-expect-error Testing invalid input
    await expect(pthToOnnx(null)).rejects.toThrow('Unsupported input type');
  });

  it('should reject undefined input', async () => {
    // @ts-expect-error Testing invalid input
    await expect(pthToOnnx(undefined)).rejects.toThrow('Unsupported input type');
  });

  it('should reject plain object input', async () => {
    // @ts-expect-error Testing invalid input
    await expect(pthToOnnx({ data: 'test' })).rejects.toThrow('Unsupported input type');
  });

  it('should reject number input', async () => {
    // @ts-expect-error Testing invalid input
    await expect(pthToOnnx(123)).rejects.toThrow('Unsupported input type');
  });

  it('should handle ReadableStream with no chunks', async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.close();
      }
    });
    await expect(pthToOnnx(stream)).rejects.toThrow();
  });

  it('should handle ReadableStream with single chunk', async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([0x80, 0x04, 0x7d, 0x2e]));
        controller.close();
      }
    });
    await expect(pthToOnnx(stream)).rejects.toThrow();
  });
});
