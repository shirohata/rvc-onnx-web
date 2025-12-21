
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { parsePth, detectWeightNorm, detectSpectralNorm } from '../src/pth-parser';
import * as fflate from 'fflate';
import { TensorData } from '../src/types';

// Mock fflate
vi.mock('fflate', () => ({
  unzipSync: vi.fn(),
}));

// Mock Unpickler
vi.mock('../src/pickle', () => {
  return {
    Unpickler: vi.fn().mockImplementation(() => ({
      load: vi.fn().mockReturnValue({
        config: [],
        weight: {},
        f0: 1,
        version: "v2",
        vocoder: "hifigan"
      })
    })),
    TorchStorage: class {}
  };
});

describe('PTH Parser Extended Coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should throw error for non-zip files (legacy format)', async () => {
    const buffer = new Uint8Array([0x00, 0x01, 0x02]).buffer;
    await expect(parsePth(buffer)).rejects.toThrow("Legacy pickle format not supported");
  });

  it('should throw error if no pickle file found in zip', async () => {
    const buffer = new Uint8Array([0x50, 0x4b, 0x00]).buffer; // PK header
    vi.mocked(fflate.unzipSync).mockReturnValue({
      'other.txt': new Uint8Array([1])
    });

    await expect(parsePth(buffer)).rejects.toThrow("No pickle file found");
  });

  it('should detect legacy weight norm', () => {
    const weights = new Map<string, TensorData>();
    weights.set('layer.weight_g', { data: new Float32Array(1), shape: [1], dtype: 'float32' });
    weights.set('layer.weight_v', { data: new Float32Array(1), shape: [1], dtype: 'float32' });
    
    const params = detectWeightNorm(weights);
    expect(params).toHaveLength(1);
    expect(params[0].baseName).toBe('layer');
    expect(params[0].weightG).toBe('layer.weight_g');
  });

  it('should detect spectral norm', () => {
    const weights = new Map<string, TensorData>();
    weights.set('layer.weight_orig', { data: new Float32Array(1), shape: [1], dtype: 'float32' });
    weights.set('layer.weight_u', { data: new Float32Array(1), shape: [1], dtype: 'float32' });
    weights.set('layer.weight_v', { data: new Float32Array(1), shape: [1], dtype: 'float32' });
    
    const params = detectSpectralNorm(weights);
    expect(params).toHaveLength(1);
    expect(params[0].baseName).toBe('layer');
    expect(params[0].weightOrig).toBe('layer.weight_orig');
  });
});

