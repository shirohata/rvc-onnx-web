
import { describe, it, expect } from 'vitest';
import { buildOnnxModel } from '../src/onnx-builder';
import { ParsedCheckpoint, TensorData } from '../src/types';

describe('Synthesizer Builder Branch Coverage', () => {
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
    resblockKernelSizes: [3],
    resblockDilationSizes: [[1]],
    upsampleRates: [2],
    upsampleInitialChannel: 10,
    upsampleKernelSizes: [4],
    spkEmbedDim: 109,
    ginChannels: 256,
    sr: 40000
  };

  const createTensor = (shape: number[]): TensorData => ({
    data: new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(0.1),
    shape,
    dtype: 'float32'
  });

  const createProxyWeights = (
    baseWeights: Map<string, TensorData>,
    hasCallback: (key: string) => boolean
  ) => {
    return new Proxy(baseWeights, {
      get(target, prop) {
        if (prop === 'get') {
          return (key: string) => {
            if (target.has(key)) return target.get(key);
            
            // Handle relative position embedding shape
            if (key.includes('emb_rel_k') || key.includes('emb_rel_v')) {
              // [n_heads, 2*window_size+1, k_channels]
              // window_size=4 -> 9
              return createTensor([2, 9, 5]); 
            }
            
            // Return dummy for others
            return createTensor([1]); 
          };
        }
        if (prop === 'has') {
          return hasCallback;
        }
        return Reflect.get(target, prop);
      }
    }) as Map<string, TensorData>;
  };

  it('should handle legacy weight norm format', () => {
    const weights = new Map<string, TensorData>();
    
    // Add essential weights
    weights.set('emb_g.weight', createTensor([10, 256]));
    weights.set('enc_p.emb_phone.weight', createTensor([10, 10]));
    weights.set('enc_p.emb_phone.bias', createTensor([10]));
    weights.set('enc_p.proj.weight', createTensor([20, 10, 1]));
    weights.set('enc_p.proj.bias', createTensor([20]));
    
    // Add flow weights with LEGACY weight norm
    // flow.flows.0.enc.in_layers.0.weight_g
    weights.set('flow.flows.0.enc.in_layers.0.weight_g', createTensor([20, 1, 1]));
    weights.set('flow.flows.0.enc.in_layers.0.weight_v', createTensor([20, 10, 3]));
    weights.set('flow.flows.0.enc.in_layers.0.bias', createTensor([20]));
    
    const proxyWeights = createProxyWeights(weights, (key: string) => {
      if (weights.has(key)) return true;
      // Force "modern" keys to be missing to trigger legacy check
      if (key.includes('parametrizations')) return false;
      // Force "legacy" keys to be present if we set them
      if (key.includes('weight_g') || key.includes('weight_v')) return weights.has(key);
      
      // For everything else, say it exists so we don't crash
      return true;
    });

    const checkpoint: ParsedCheckpoint = {
      config: baseConfig,
      weights: proxyWeights,
      useF0: false,
      version: "v1",
      vocoder: "hifigan"
    };

    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 10 });
    expect(model).toBeDefined();
  });

  it('should handle missing biases (optional bias branches)', () => {
    const weights = new Map<string, TensorData>();
    
    const proxyWeights = createProxyWeights(weights, (key: string) => {
      // Force biases to be missing
      if (key.endsWith('.bias')) return false;
      return true;
    });

    const checkpoint: ParsedCheckpoint = {
      config: baseConfig,
      weights: proxyWeights,
      useF0: false,
      version: "v1",
      vocoder: "hifigan"
    };

    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 10 });
    expect(model).toBeDefined();
  });

  it('should handle no weight norm (standard weights)', () => {
    const weights = new Map<string, TensorData>();
    
    const proxyWeights = createProxyWeights(weights, (key: string) => {
      // Force weight norm keys to be missing
      if (key.includes('parametrizations') || key.includes('weight_g') || key.includes('weight_v')) return false;
      return true;
    });

    const checkpoint: ParsedCheckpoint = {
      config: baseConfig,
      weights: proxyWeights,
      useF0: false,
      version: "v1",
      vocoder: "hifigan"
    };

    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 10 });
    expect(model).toBeDefined();
  });
});
