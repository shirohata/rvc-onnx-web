/**
 * Browser-based RVC .pth -> ONNX converter
 *
 * This module provides the main entry point for converting PyTorch
 * checkpoint files (.pth) to ONNX format entirely in the browser.
 */

export { pthToOnnx, type ConversionOptions, type ConversionResult, type PthInput } from "./converter.js";
export { parsePth, type PthCheckpoint } from "./pth-parser.js";
export { buildOnnxModel } from "./onnx-builder.js";
export { serializeOnnx } from "./onnx-serializer.js";
export { Unpickler, PythonObject, type TorchStorage, type DType } from "./pickle.js";
export type { 
  TensorData, 
  RvcConfig, 
  ParsedCheckpoint,
  OnnxModel,
  OnnxGraph,
  OnnxNode
} from "./types.js";
export { OnnxDataType } from "./types.js";
