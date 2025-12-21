/**
 * Browser-based RVC .pth -> ONNX converter
 *
 * This module provides the main entry point for converting PyTorch
 * checkpoint files (.pth) to ONNX format entirely in the browser.
 */

export { pthToOnnx, type ConversionOptions, type ConversionResult, type PthInput } from "./converter";
export { parsePth, type PthCheckpoint } from "./pth-parser";
export { buildOnnxModel } from "./onnx-builder";
export { serializeOnnx } from "./onnx-serializer";
export { Unpickler, PythonObject, type TorchStorage, type DType } from "./pickle";
export type { 
  TensorData, 
  RvcConfig, 
  ParsedCheckpoint,
  OnnxModel,
  OnnxGraph,
  OnnxNode
} from "./types";
export { OnnxDataType } from "./types";
