/**
 * PyTorch .pth file parser.
 *
 * This module parses PyTorch checkpoint files (.pth/.pt) which are ZIP archives
 * containing pickled Python objects and raw tensor data.
 *
 * Format Overview:
 * - Modern PyTorch files (v1.6+) are ZIP archives
 * - Contains: data.pkl (pickled dict) and data/0, data/1, ... (raw tensors)
 * - The pickle references "storage" objects that map to the binary files
 */

import * as fflate from "fflate";
import {
  ParsedCheckpoint,
  RvcConfig,
  TensorData,
} from "./types";
import { Unpickler, TorchStorage } from "./pickle";

/**
 * Represents the raw structure of a PyTorch checkpoint.
 */
export interface PthCheckpoint {
  config: RvcConfig;
  weights: Map<string, TensorData>;
  useF0: boolean;
  version: "v1" | "v2";
  vocoder: string;
}

/**
 * Parse a .pth file buffer and extract weights and configuration.
 *
 * @param buffer - The raw .pth file contents
 * @returns Parsed checkpoint with weights and config
 */
export async function parsePth(buffer: ArrayBuffer): Promise<ParsedCheckpoint> {
  const bytes = new Uint8Array(buffer);

  // Check if it's a ZIP file (modern PyTorch format)
  const isZip = bytes[0] === 0x50 && bytes[1] === 0x4b; // "PK"

  if (!isZip) {
    throw new Error(
      "Legacy pickle format not supported. Please use a PyTorch 1.6+ checkpoint."
    );
  }

  // Unzip the archive
  const unzipped = fflate.unzipSync(bytes);

  // Find the pickle file (usually "data.pkl" or "archive/data.pkl")
  let pickleData: Uint8Array | undefined;
  let dataPrefix = "";

  for (const [path, data] of Object.entries(unzipped)) {
    if (path.endsWith("data.pkl") || path.endsWith(".pkl")) {
      pickleData = data;
      // Determine the prefix for tensor data files
      // e.g., if pickle is at "archive/data.pkl", tensors are at "archive/data/0"
      const dir = path.substring(0, path.lastIndexOf("/") + 1);
      dataPrefix = dir + "data/";
      break;
    }
  }

  if (!pickleData) {
    throw new Error("No pickle file found in .pth archive");
  }

  // Create a storage resolver that maps storage keys to binary data
  const storageResolver = (key: string): Uint8Array => {
    // Try different possible paths for the tensor data
    const possiblePaths = [
      `${dataPrefix}${key}`,
      `data/${key}`,
      `archive/data/${key}`,
      key,
    ];

    for (const path of possiblePaths) {
      if (unzipped[path]) {
        return unzipped[path];
      }
    }

    throw new Error(`Storage key not found: ${key}`);
  };

  // Unpickle the checkpoint
  const unpickler = new Unpickler(pickleData, storageResolver);
  const checkpoint = unpickler.load();

  // Validate checkpoint structure
  if (!checkpoint.config || !checkpoint.weight) {
    throw new Error(
      "Invalid checkpoint: missing 'config' or 'weight' keys. " +
        "This may not be an RVC model."
    );
  }

  // Parse the config array into a structured object
  const configArray = checkpoint.config as unknown[];
  const config = parseConfigArray(configArray);

  // Convert weights to TensorData map
  const weights = new Map<string, TensorData>();
  const weightDict = checkpoint.weight as Record<string, TorchStorage>;

  // Helper to determine actual dtype from data type
  const getActualDtype = (data: TorchStorage["data"], originalDtype?: string): TensorData["dtype"] => {
    // The pickle parser converts float16/bfloat16/float64 to Float32Array,
    // so we should use the actual data type, not the original storage type
    if (data instanceof Float32Array) return "float32";
    if (data instanceof BigInt64Array) return "int64";
    if (data instanceof Int32Array) return "int32";
    if (data instanceof Uint8Array) return "uint8";
    if (data instanceof Float64Array) return "float64";
    // Fallback to original dtype if available
    return (originalDtype as TensorData["dtype"]) || "float32";
  };

  for (const [name, storage] of Object.entries(weightDict)) {
    if (storage && storage.data) {
      weights.set(name, {
        data: storage.data,
        shape: storage.shape || [],
        dtype: getActualDtype(storage.data, storage.dtype),
        requiresGrad: storage.requiresGrad,
      });
    }
  }

  // Extract metadata
  const useF0 = Boolean(checkpoint.f0 ?? 1);
  const version = (checkpoint.version as "v1" | "v2") || "v1";
  const vocoder = (checkpoint.vocoder as string) || "HiFi-GAN";

  // Update spk_embed_dim from weights if available
  if (weights.has("emb_g.weight")) {
    const embWeight = weights.get("emb_g.weight")!;
    config.spkEmbedDim = embWeight.shape[0];
  }

  return {
    config,
    weights,
    useF0,
    version,
    vocoder,
  };
}

/**
 * Parse the RVC config array into a structured object.
 */
function parseConfigArray(arr: unknown[]): RvcConfig {
  // [spec_channels, segment_size, inter_channels, hidden_channels,
  //  filter_channels, n_heads, n_layers, kernel_size, p_dropout,
  //  resblock, resblock_kernel_sizes, resblock_dilation_sizes,
  //  upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
  //  spk_embed_dim, gin_channels, sr]

  return {
    specChannels: arr[0] as number,
    segmentSize: arr[1] as number,
    interChannels: arr[2] as number,
    hiddenChannels: arr[3] as number,
    filterChannels: arr[4] as number,
    nHeads: arr[5] as number,
    nLayers: arr[6] as number,
    kernelSize: arr[7] as number,
    pDropout: arr[8] as number,
    resblock: arr[9] as string,
    resblockKernelSizes: arr[10] as number[],
    resblockDilationSizes: arr[11] as number[][],
    upsampleRates: arr[12] as number[],
    upsampleInitialChannel: arr[13] as number,
    upsampleKernelSizes: arr[14] as number[],
    spkEmbedDim: arr[15] as number,
    ginChannels: arr[16] as number,
    sr: arr[17] as number,
  };
}

// =============================================================================
// Weight Handling Utilities
// =============================================================================

/**
 * Weight normalization parameter detection.
 * 
 * PyTorch weight_norm stores weights as:
 *   - layer.parametrizations.weight.original0 (weight_g - magnitude)
 *   - layer.parametrizations.weight.original1 (weight_v - direction)
 * 
 * Combined weight: weight = weight_g * (weight_v / ||weight_v||)
 * 
 * Note: Older PyTorch used layer.weight_g and layer.weight_v directly.
 */
export interface WeightNormParams {
  weightG: string;  // Magnitude tensor name
  weightV: string;  // Direction tensor name
  baseName: string; // Original layer name (e.g., "enc.in_layers.0")
}

/**
 * Detect weight_norm parameters in weights map.
 * 
 * @param weights - Map of weight names to tensor data
 * @returns Array of detected weight_norm parameter groups
 */
export function detectWeightNorm(weights: Map<string, TensorData>): WeightNormParams[] {
  const results: WeightNormParams[] = [];
  const processed = new Set<string>();
  
  for (const name of weights.keys()) {
    // Modern PyTorch parametrization format
    // e.g., "enc.in_layers.0.parametrizations.weight.original0"
    const modernMatch = name.match(/^(.+)\.parametrizations\.weight\.original0$/);
    if (modernMatch) {
      const baseName = modernMatch[1];
      const weightV = `${baseName}.parametrizations.weight.original1`;
      
      if (weights.has(weightV) && !processed.has(baseName)) {
        results.push({
          weightG: name,
          weightV: weightV,
          baseName: baseName,
        });
        processed.add(baseName);
      }
    }
    
    // Legacy format: layer.weight_g, layer.weight_v
    const legacyMatch = name.match(/^(.+)\.weight_g$/);
    if (legacyMatch) {
      const baseName = legacyMatch[1];
      const weightV = `${baseName}.weight_v`;
      
      if (weights.has(weightV) && !processed.has(baseName)) {
        results.push({
          weightG: name,
          weightV: weightV,
          baseName: baseName,
        });
        processed.add(baseName);
      }
    }
  }
  
  return results;
}

/**
 * Spectral normalization parameter detection.
 * 
 * PyTorch spectral_norm stores:
 *   - layer.weight_u: Left singular vector (updated during training)
 *   - layer.weight_v: Right singular vector (updated during training)
 *   - layer.weight_orig: Original weight matrix
 * 
 * At inference, the normalized weight is computed as:
 *   weight = weight_orig / sigma
 *   where sigma = u^T @ W @ v (largest singular value estimate)
 * 
 * For ONNX export, we can either:
 *   1. Pre-compute the normalized weight (recommended)
 *   2. Include the normalization in the graph (more accurate but complex)
 */
export interface SpectralNormParams {
  weightU: string;     // Left singular vector
  weightV: string;     // Right singular vector  
  weightOrig: string;  // Original weight
  baseName: string;    // Layer name
}

/**
 * Detect spectral_norm parameters in weights map.
 */
export function detectSpectralNorm(weights: Map<string, TensorData>): SpectralNormParams[] {
  const results: SpectralNormParams[] = [];
  const processed = new Set<string>();
  
  for (const name of weights.keys()) {
    // Format: layer.weight_orig (with layer.weight_u and layer.weight_v)
    const match = name.match(/^(.+)\.weight_orig$/);
    if (match) {
      const baseName = match[1];
      const weightU = `${baseName}.weight_u`;
      const weightV = `${baseName}.weight_v`;
      
      if (weights.has(weightU) && weights.has(weightV) && !processed.has(baseName)) {
        results.push({
          weightU: weightU,
          weightV: weightV,
          weightOrig: name,
          baseName: baseName,
        });
        processed.add(baseName);
      }
    }
  }
  
  return results;
}

/**
 * Check if a weight tensor is zero-initialized.
 * 
 * Some layers in RVC are explicitly zero-initialized:
 *   - Flow's post convolution (for identity initialization)
 *   - Some projection layers
 * 
 * @param tensor - Tensor data to check
 * @param tolerance - Maximum absolute value to consider as "zero"
 * @returns True if all values are within tolerance of zero
 */
export function isZeroInitialized(tensor: TensorData, tolerance: number = 1e-7): boolean {
  const data = tensor.data;
  
  if (data instanceof Float32Array) {
    for (let i = 0; i < data.length; i++) {
      if (Math.abs(data[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }
  
  // For other types, convert and check
  for (let i = 0; i < data.length; i++) {
    const val = Number(data[i]);
    if (Math.abs(val) > tolerance) {
      return false;
    }
  }
  return true;
}

/**
 * Get a summary of weight normalization usage in the model.
 */
export function getWeightNormSummary(weights: Map<string, TensorData>): {
  weightNormCount: number;
  spectralNormCount: number;
  zeroInitializedLayers: string[];
  weightNormLayers: string[];
  spectralNormLayers: string[];
} {
  const weightNorm = detectWeightNorm(weights);
  const spectralNorm = detectSpectralNorm(weights);
  
  const zeroInitializedLayers: string[] = [];
  
  // Check known zero-initialized layers
  const zeroInitCandidates = [
    "flow.flows.0.post",
    "flow.flows.2.post", 
    "flow.flows.4.post",
    "flow.flows.6.post",
  ];
  
  for (const layer of zeroInitCandidates) {
    const weightName = `${layer}.weight`;
    const biasName = `${layer}.bias`;
    
    const weight = weights.get(weightName);
    const bias = weights.get(biasName);
    
    if (weight && isZeroInitialized(weight)) {
      zeroInitializedLayers.push(layer);
    } else if (bias && isZeroInitialized(bias)) {
      zeroInitializedLayers.push(`${layer} (bias only)`);
    }
  }
  
  return {
    weightNormCount: weightNorm.length,
    spectralNormCount: spectralNorm.length,
    zeroInitializedLayers,
    weightNormLayers: weightNorm.map(w => w.baseName),
    spectralNormLayers: spectralNorm.map(s => s.baseName),
  };
}
