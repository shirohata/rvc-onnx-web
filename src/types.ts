/**
 * Core type definitions for the PTH to ONNX converter.
 */

import type { DType } from "./pickle.js";

/**
 * Represents a tensor extracted from a .pth file.
 */
export interface TensorData {
  /** The raw float data */
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array | Int16Array;
  /** Shape of the tensor [dim0, dim1, ...] */
  shape: number[];
  /** Data type string */
  dtype: DType;
  /** Whether this tensor requires gradient (from training) */
  requiresGrad?: boolean;
}

/**
 * [spec_channels, segment_size, inter_channels, hidden_channels,
 *  filter_channels, n_heads, n_layers, kernel_size, p_dropout,
 *  resblock, resblock_kernel_sizes, resblock_dilation_sizes,
 *  upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
 *  spk_embed_dim, gin_channels, sr]
 */
export interface RvcConfig {
  specChannels: number;
  segmentSize: number;
  interChannels: number;
  hiddenChannels: number;
  filterChannels: number;
  nHeads: number;
  nLayers: number;
  kernelSize: number;
  pDropout: number;
  resblock: string;
  resblockKernelSizes: number[];
  resblockDilationSizes: number[][];
  upsampleRates: number[];
  upsampleInitialChannel: number;
  upsampleKernelSizes: number[];
  spkEmbedDim: number;
  ginChannels: number;
  sr: number;
}

/**
 * Parsed checkpoint structure from .pth file.
 */
export interface ParsedCheckpoint {
  /** Model configuration */
  config: RvcConfig;
  /** Weight tensors keyed by parameter name */
  weights: Map<string, TensorData>;
  /** Whether the model uses F0 (pitch) information */
  useF0: boolean;
  /** Model version ("v1" or "v2") */
  version: "v1" | "v2";
  /** Vocoder type */
  vocoder: string;
}

/**
 * ONNX tensor element types (from ONNX spec).
 */
export enum OnnxDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
}

/**
 * An ONNX graph node (operator).
 */
export interface OnnxNode {
  /** Operator type (e.g., "Conv", "MatMul", "Relu") */
  opType: string;
  /** Node name (for debugging) */
  name: string;
  /** Input tensor names */
  inputs: string[];
  /** Output tensor names */
  outputs: string[];
  /** Operator attributes */
  attributes: OnnxAttribute[];
}

/**
 * An ONNX node attribute.
 */
export interface OnnxAttribute {
  name: string;
  type: "INT" | "INTS" | "FLOAT" | "FLOATS" | "STRING" | "TENSOR";
  intValue?: bigint;
  intsValue?: bigint[];
  floatValue?: number;
  floatsValue?: number[];
  stringValue?: string;
  tensorValue?: TensorData;
}

/**
 * ONNX tensor shape dimension.
 */
export interface OnnxDimension {
  /** Static dimension value (if known) */
  dimValue?: bigint;
  /** Dynamic dimension name (if symbolic) */
  dimParam?: string;
}

/**
 * ONNX value info (describes an input/output).
 */
export interface OnnxValueInfo {
  name: string;
  elemType: OnnxDataType;
  shape: OnnxDimension[];
}

/**
 * ONNX initializer (constant tensor).
 */
export interface OnnxInitializer {
  name: string;
  data: TensorData;
}

/**
 * Complete ONNX graph structure.
 */
export interface OnnxGraph {
  name: string;
  nodes: OnnxNode[];
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializers: OnnxInitializer[];
}

/**
 * Complete ONNX model.
 */
export interface OnnxModel {
  irVersion: bigint;
  opsetImports: Array<{ domain: string; version: bigint }>;
  producerName: string;
  producerVersion: string;
  graph: OnnxGraph;
}
