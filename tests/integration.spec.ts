
import { describe, it, expect } from 'vitest';
import { buildOnnxModel } from '../src/onnx-builder';
import { ParsedCheckpoint, TensorData } from '../src/types';

describe('Integration: ONNX Builder Edge Cases', () => {
  const createMockWeights = (): Map<string, TensorData> => {
    return new Proxy(new Map(), {
      get(target, prop) {
        if (prop === 'get') {
          return (key: string) => {
            // Return a dummy tensor
            // We use a shape that is likely to be compatible with most operations
            // [1, 1, 1] is safe for simple ops, but for convolutions we might need more
            // Let's try a generic shape.
            return {
              data: new Float32Array(1024).fill(0.1), 
              shape: [1, 1, 1024], 
              dtype: 'float32'
            };
          };
        }
        if (prop === 'has') return () => true;
        return Reflect.get(target, prop);
      }
    }) as Map<string, TensorData>;
  };

  const baseConfig = {
    specChannels: 10,
    segmentSize: 32,
    interChannels: 10,
    hiddenChannels: 10,
    filterChannels: 10,
    nHeads: 2,
    nLayers: 2,
    kernelSize: 3,
    pDropout: 0.1,
    resblock: "1",
    resblockKernelSizes: [3, 7, 11],
    resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsampleRates: [8, 8, 2, 2],
    upsampleInitialChannel: 512,
    upsampleKernelSizes: [16, 16, 4, 4],
    spkEmbedDim: 109,
    ginChannels: 256,
    sr: 40000
  };

  it('should build graph for v1 model without F0', () => {
    const checkpoint: ParsedCheckpoint = {
      config: baseConfig,
      weights: createMockWeights(),
      useF0: false,
      version: "v1",
      vocoder: "hifigan"
    };

    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 256 });
    expect(model).toBeDefined();
    expect(model.graph).toBeDefined();
    expect(model.graph.nodes.length).toBeGreaterThan(0);
  });

  it('should build graph for v2 model with F0', () => {
    const checkpoint: ParsedCheckpoint = {
      config: baseConfig,
      weights: createMockWeights(),
      useF0: true,
      version: "v2",
      vocoder: "hifigan"
    };

    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 256 });
    expect(model).toBeDefined();
    expect(model.graph).toBeDefined();
    expect(model.graph.nodes.length).toBeGreaterThan(0);
  });
});
