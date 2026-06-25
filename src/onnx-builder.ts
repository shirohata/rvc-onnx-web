/**
 * ONNX model builder - constructs ONNX graph from parsed checkpoint.
 *
 * This module builds the complete ONNX graph structure by translating
 * the RVC Synthesizer architecture into ONNX nodes.
 */

import type {
  ParsedCheckpoint,
  OnnxModel,
  OnnxNode,
  OnnxValueInfo,
  OnnxInitializer,
  OnnxDataType,
  OnnxAttribute,
  TensorData,
} from "./types.js";
import { buildSynthesizerGraph } from "./synthesizer-builder.js";

export type ExportMode = "default" | "webui";
export type TargetRuntime = "default" | "tensorrt";

export interface BuildOptions {
  opsetVersion: number;
  phoneLen: number;
  exportMode?: ExportMode;
  targetRuntime?: TargetRuntime;
}

/**
 * Build a complete ONNX model from a parsed checkpoint.
 */
export function buildOnnxModel(
  checkpoint: ParsedCheckpoint,
  options: BuildOptions
): OnnxModel {
  const {
    opsetVersion,
    phoneLen,
    exportMode = "default",
    targetRuntime = "default",
  } = options;

  // Build the synthesizer graph
  const graph = buildSynthesizerGraph(checkpoint, phoneLen, {
    exportMode,
    targetRuntime,
  });

  return {
    irVersion: 8n, // ONNX IR version 8
    opsetImports: [
      { domain: "", version: BigInt(opsetVersion) }, // Default ONNX domain
    ],
    producerName: "browser-pth-to-onnx",
    producerVersion: "1.0.0",
    graph,
  };
}

// =============================================================================
// Graph Building Utilities
// =============================================================================

/**
 * Counter for generating unique names.
 */
let _nameCounter = 0;

/**
 * Reset the name counter (for testing).
 */
export function resetNameCounter(): void {
  _nameCounter = 0;
}

/**
 * Generate a unique name for a node or tensor.
 */
export function uniqueName(prefix: string): string {
  return `${prefix}_${_nameCounter++}`;
}

/**
 * Create an ONNX attribute for an integer value.
 */
export function attrInt(name: string, value: number | bigint): OnnxAttribute {
  return {
    name,
    type: "INT",
    intValue: BigInt(value),
  };
}

/**
 * Create an ONNX attribute for an array of integers.
 */
export function attrInts(name: string, values: number[]): OnnxAttribute {
  return {
    name,
    type: "INTS",
    intsValue: values.map((v) => BigInt(v)),
  };
}

/**
 * Create an ONNX attribute for a float value.
 */
export function attrFloat(name: string, value: number): OnnxAttribute {
  return {
    name,
    type: "FLOAT",
    floatValue: value,
  };
}

/**
 * Create an ONNX attribute for an array of floats.
 */
export function attrFloats(name: string, values: number[]): OnnxAttribute {
  return {
    name,
    type: "FLOATS",
    floatsValue: values,
  };
}

/**
 * Create an ONNX attribute for a string value.
 */
export function attrString(name: string, value: string): OnnxAttribute {
  return {
    name,
    type: "STRING",
    stringValue: value,
  };
}

/**
 * Create an ONNX value info (for inputs/outputs).
 */
export function valueInfo(
  name: string,
  elemType: OnnxDataType,
  shape: (number | string)[]
): OnnxValueInfo {
  return {
    name,
    elemType,
    shape: shape.map((dim) =>
      typeof dim === "number"
        ? { dimValue: BigInt(dim) }
        : { dimParam: dim }
    ),
  };
}

/**
 * Create an ONNX initializer from tensor data.
 */
export function initializer(name: string, tensor: TensorData): OnnxInitializer {
  return {
    name,
    data: tensor,
  };
}

/**
 * Create an ONNX node.
 */
export function node(
  opType: string,
  inputs: string[],
  outputs: string[],
  attributes: OnnxAttribute[] = [],
  name?: string
): OnnxNode {
  return {
    opType,
    name: name || uniqueName(opType.toLowerCase()),
    inputs,
    outputs,
    attributes,
  };
}

// =============================================================================
// Common Layer Builders
// =============================================================================

/**
 * Build a Conv1d node.
 * ONNX Conv expects weight shape: [out_channels, in_channels/groups, kernel_size]
 */
export function conv1d(
  input: string,
  weight: string,
  bias: string | null,
  output: string,
  kernelSize: number,
  stride: number = 1,
  padding: number = 0,
  dilation: number = 1,
  groups: number = 1
): OnnxNode {
  const inputs = bias ? [input, weight, bias] : [input, weight];
  return node("Conv", inputs, [output], [
    attrInts("kernel_shape", [kernelSize]),
    attrInts("strides", [stride]),
    attrInts("pads", [padding, padding]),
    attrInts("dilations", [dilation]),
    attrInt("group", groups),
  ]);
}

/**
 * Build a ConvTranspose1d node.
 */
export function convTranspose1d(
  input: string,
  weight: string,
  bias: string | null,
  output: string,
  kernelSize: number,
  stride: number = 1,
  padding: number = 0,
  outputPadding: number = 0,
  dilation: number = 1,
  groups: number = 1
): OnnxNode {
  const inputs = bias ? [input, weight, bias] : [input, weight];
  return node("ConvTranspose", inputs, [output], [
    attrInts("kernel_shape", [kernelSize]),
    attrInts("strides", [stride]),
    attrInts("pads", [padding, padding]),
    attrInts("output_padding", [outputPadding]),
    attrInts("dilations", [dilation]),
    attrInt("group", groups),
  ]);
}

/**
 * Build a Linear layer using MatMul + Add.
 * 
 * Unlike Gemm, MatMul supports arbitrary ranks (e.g., [B, T, H] @ [H, K] -> [B, T, K]).
 * Since PyTorch stores linear weights as [out_features, in_features], we transpose them.
 * 
 * Returns an array of nodes: [Transpose, MatMul, (optional) Add]
 */
export function linearNodes(
  input: string,
  weight: string,
  bias: string | null,
  output: string
): OnnxNode[] {
  const nodes: OnnxNode[] = [];
  
  // Transpose weight from [out, in] to [in, out]
  const weightT = uniqueName("weight_transposed");
  nodes.push(node("Transpose", [weight], [weightT], [attrInts("perm", [1, 0])]));
  
  // MatMul: input @ weight^T
  const matmulOut = bias ? uniqueName("matmul_out") : output;
  nodes.push(node("MatMul", [input, weightT], [matmulOut]));
  
  // Add bias if present
  if (bias) {
    nodes.push(node("Add", [matmulOut, bias], [output]));
  }
  
  return nodes;
}

/**
 * Build a Linear layer (legacy single-node interface for compatibility).
 * 
 * NOTE: This creates multiple nodes internally. For proper graph building,
 * use linearNodes() which returns all required nodes.
 * 
 * @deprecated Use linearNodes() instead for proper multi-node handling.
 */
export function linear(
  input: string,
  weight: string,
  bias: string | null,
  output: string
): OnnxNode {
  // Return first node - caller should use linearNodes() for full support
  console.warn("linear() is deprecated, use linearNodes() for rank > 2 support");
  const inputs = bias ? [input, weight, bias] : [input, weight];
  return node("Gemm", inputs, [output], [
    attrInt("transB", 1),
  ]);
}

/**
 * Build a MatMul node.
 */
export function matmul(
  a: string,
  b: string,
  output: string
): OnnxNode {
  return node("MatMul", [a, b], [output]);
}

/**
 * Build a LayerNorm node (ONNX 17+).
 */
export function layerNorm(
  input: string,
  scale: string,
  bias: string,
  output: string,
  axis: number = -1,
  epsilon: number = 1e-5
): OnnxNode {
  return node("LayerNormalization", [input, scale, bias], [output], [
    attrInt("axis", axis),
    attrFloat("epsilon", epsilon),
  ]);
}

/**
 * Build a Relu node.
 */
export function relu(input: string, output: string): OnnxNode {
  return node("Relu", [input], [output]);
}

/**
 * Build a LeakyRelu node.
 */
export function leakyRelu(
  input: string,
  output: string,
  alpha: number = 0.01
): OnnxNode {
  return node("LeakyRelu", [input], [output], [attrFloat("alpha", alpha)]);
}

/**
 * Build a Sigmoid node.
 */
export function sigmoid(input: string, output: string): OnnxNode {
  return node("Sigmoid", [input], [output]);
}

/**
 * Build a Tanh node.
 */
export function tanh(input: string, output: string): OnnxNode {
  return node("Tanh", [input], [output]);
}

/**
 * Build an Add node.
 */
export function add(a: string, b: string, output: string): OnnxNode {
  return node("Add", [a, b], [output]);
}

/**
 * Build a Mul node.
 */
export function mul(a: string, b: string, output: string): OnnxNode {
  return node("Mul", [a, b], [output]);
}

/**
 * Build a Sub node.
 */
export function sub(a: string, b: string, output: string): OnnxNode {
  return node("Sub", [a, b], [output]);
}

/**
 * Build a Div node.
 */
export function div(a: string, b: string, output: string): OnnxNode {
  return node("Div", [a, b], [output]);
}

/**
 * Build an Exp node.
 */
export function exp(input: string, output: string): OnnxNode {
  return node("Exp", [input], [output]);
}

/**
 * Build a Softmax node.
 */
export function softmax(
  input: string,
  output: string,
  axis: number = -1
): OnnxNode {
  return node("Softmax", [input], [output], [attrInt("axis", axis)]);
}

/**
 * Build a Transpose node.
 */
export function transpose(
  input: string,
  output: string,
  perm: number[]
): OnnxNode {
  return node("Transpose", [input], [output], [attrInts("perm", perm)]);
}

/**
 * Build a Reshape node.
 */
export function reshape(
  input: string,
  shape: string,
  output: string,
  allowzero: number = 0
): OnnxNode {
  return node("Reshape", [input, shape], [output], [
    attrInt("allowzero", allowzero),
  ]);
}

/**
 * Build a Squeeze node.
 */
export function squeeze(
  input: string,
  axes: string,
  output: string
): OnnxNode {
  return node("Squeeze", [input, axes], [output]);
}

/**
 * Build an Unsqueeze node.
 */
export function unsqueeze(
  input: string,
  axes: string,
  output: string
): OnnxNode {
  return node("Unsqueeze", [input, axes], [output]);
}

/**
 * Build a Concat node.
 */
export function concat(
  inputs: string[],
  output: string,
  axis: number = 0
): OnnxNode {
  return node("Concat", inputs, [output], [attrInt("axis", axis)]);
}

/**
 * Build a Split node.
 * 
 * In ONNX opset 13+, Split requires a 'split' input instead of 'num_outputs' attribute.
 * The split input specifies the size of each output along the split axis.
 * If splitSizes is not provided, the input is split equally into len(outputs) parts.
 */
export function split(
  input: string,
  outputs: string[],
  axis: number = 0,
  splitSizesInput?: string  // Name of the split sizes tensor (int64)
): OnnxNode {
  const attrs = [attrInt("axis", axis)];
  const inputs = splitSizesInput ? [input, splitSizesInput] : [input];
  return node("Split", inputs, outputs, attrs);
}

/**
 * Build a Gather node (for embeddings).
 */
export function gather(
  input: string,
  indices: string,
  output: string,
  axis: number = 0
): OnnxNode {
  return node("Gather", [input, indices], [output], [attrInt("axis", axis)]);
}

/**
 * Build a Slice node.
 */
export function slice(
  input: string,
  starts: string,
  ends: string,
  axes: string,
  steps: string,
  output: string
): OnnxNode {
  return node("Slice", [input, starts, ends, axes, steps], [output]);
}

/**
 * Build a ReduceMean node.
 */
export function reduceMean(
  input: string,
  output: string,
  axes: number[],
  keepdims: boolean = true
): OnnxNode {
  return node("ReduceMean", [input], [output], [
    attrInts("axes", axes),
    attrInt("keepdims", keepdims ? 1 : 0),
  ]);
}

/**
 * Build a Pad node.
 */
export function pad(
  input: string,
  pads: string,
  output: string,
  mode: string = "constant",
  constantValue?: string
): OnnxNode {
  const inputs = constantValue
    ? [input, pads, constantValue]
    : [input, pads];
  return node("Pad", inputs, [output], [attrString("mode", mode)]);
}

/**
 * Build a Sin node.
 */
export function sin(input: string, output: string): OnnxNode {
  return node("Sin", [input], [output]);
}

/**
 * Build a Cos node.
 */
export function cos(input: string, output: string): OnnxNode {
  return node("Cos", [input], [output]);
}

/**
 * Build a RandomNormalLike node.
 */
export function randomNormalLike(
  input: string,
  output: string,
  mean: number = 0.0,
  scale: number = 1.0,
  dtype: number = 1 // 1 = FLOAT
): OnnxNode {
  return node("RandomNormalLike", [input], [output], [
    attrFloat("mean", mean),
    attrFloat("scale", scale),
    attrInt("dtype", dtype),
  ]);
}

// =============================================================================
// Additional Operations for Full Synthesizer Support
// =============================================================================

/**
 * Build a Neg node (element-wise negation).
 */
export function neg(input: string, output: string): OnnxNode {
  return node("Neg", [input], [output]);
}

/**
 * Build a Sqrt node.
 */
export function sqrt(input: string, output: string): OnnxNode {
  return node("Sqrt", [input], [output]);
}

/**
 * Build a Log node (natural logarithm).
 */
export function log(input: string, output: string): OnnxNode {
  return node("Log", [input], [output]);
}

/**
 * Build a Pow node.
 */
export function pow(input: string, exponent: string, output: string): OnnxNode {
  return node("Pow", [input, exponent], [output]);
}

/**
 * Build a Clip node (clamp values).
 */
export function clip(
  input: string,
  min: string | null,
  max: string | null,
  output: string
): OnnxNode {
  const inputs = [input];
  if (min) inputs.push(min);
  else inputs.push("");
  if (max) inputs.push(max);
  return node("Clip", inputs, [output]);
}

/**
 * Build a Where node (conditional selection).
 */
export function where(
  condition: string,
  x: string,
  y: string,
  output: string
): OnnxNode {
  return node("Where", [condition, x, y], [output]);
}

/**
 * Build a Less node (element-wise comparison).
 */
export function less(a: string, b: string, output: string): OnnxNode {
  return node("Less", [a, b], [output]);
}

/**
 * Build a Greater node (element-wise comparison).
 */
export function greater(a: string, b: string, output: string): OnnxNode {
  return node("Greater", [a, b], [output]);
}

/**
 * Build an Equal node (element-wise comparison).
 */
export function equal(a: string, b: string, output: string): OnnxNode {
  return node("Equal", [a, b], [output]);
}

/**
 * Build a Cast node (type conversion).
 */
export function cast(input: string, output: string, to: OnnxDataType): OnnxNode {
  return node("Cast", [input], [output], [attrInt("to", to)]);
}

/**
 * Build a Range node (generate sequence).
 */
export function range(
  start: string,
  limit: string,
  delta: string,
  output: string
): OnnxNode {
  return node("Range", [start, limit, delta], [output]);
}

/**
 * Build an Expand node (broadcast to shape).
 */
export function expand(input: string, shape: string, output: string): OnnxNode {
  return node("Expand", [input, shape], [output]);
}

/**
 * Build a Tile node (repeat tensor).
 */
export function tile(input: string, repeats: string, output: string): OnnxNode {
  return node("Tile", [input, repeats], [output]);
}

/**
 * Build a Shape node (get tensor shape).
 */
export function shape(input: string, output: string): OnnxNode {
  return node("Shape", [input], [output]);
}

/**
 * Build a ConstantOfShape node.
 */
export function constantOfShape(
  shape: string,
  output: string,
  value: number = 0,
  dtype: "float32" | "int64" = "float32"
): OnnxNode {
  const tensorValue = {
    data: dtype === "float32" ? new Float32Array([value]) : new BigInt64Array([BigInt(value)]),
    shape: [] as number[],
    dtype,
  };
  return node("ConstantOfShape", [shape], [output], [
    {
      name: "value",
      type: "TENSOR",
      tensorValue,
    },
  ]);
}

/**
 * Build an Identity node (pass-through).
 */
export function identity(input: string, output: string): OnnxNode {
  return node("Identity", [input], [output]);
}

/**
 * Build a Flatten node.
 */
export function flatten(input: string, output: string, axis: number = 1): OnnxNode {
  return node("Flatten", [input], [output], [attrInt("axis", axis)]);
}

/**
 * Build a ReduceSum node.
 * In ONNX opset 13+, axes is an optional input, not an attribute.
 * The axesInput parameter should be the name of an int64 tensor containing the axes.
 */
export function reduceSum(
  input: string,
  axesInput: string,
  output: string,
  keepdims: boolean = true,
  noop_with_empty_axes: boolean = false
): OnnxNode {
  return node("ReduceSum", [input, axesInput], [output], [
    attrInt("keepdims", keepdims ? 1 : 0),
    attrInt("noop_with_empty_axes", noop_with_empty_axes ? 1 : 0),
  ]);
}

/**
 * Build a ReduceMax node.
 * In ONNX opset 18+, axes is an optional input. For opset 17 and earlier, it's an attribute.
 */
export function reduceMax(
  input: string,
  output: string,
  axes: number[],
  keepdims: boolean = true
): OnnxNode {
  // For opset 17 and earlier, axes is an attribute
  return node("ReduceMax", [input], [output], [
    attrInts("axes", axes),
    attrInt("keepdims", keepdims ? 1 : 0),
  ]);
}

/**
 * Build a Cumsum node.
 */
export function cumsum(
  input: string,
  axis: string,
  output: string,
  exclusive: boolean = false,
  reverse: boolean = false
): OnnxNode {
  return node("CumSum", [input, axis], [output], [
    attrInt("exclusive", exclusive ? 1 : 0),
    attrInt("reverse", reverse ? 1 : 0),
  ]);
}

/**
 * Build a Mod node (modulo).
 */
export function mod(a: string, b: string, output: string, fmod: boolean = true): OnnxNode {
  return node("Mod", [a, b], [output], [attrInt("fmod", fmod ? 1 : 0)]);
}

/**
 * Build a Floor node.
 */
export function floor(input: string, output: string): OnnxNode {
  return node("Floor", [input], [output]);
}

/**
 * Build a Ceil node.
 */
export function ceil(input: string, output: string): OnnxNode {
  return node("Ceil", [input], [output]);
}

/**
 * Build an Abs node.
 */
export function abs(input: string, output: string): OnnxNode {
  return node("Abs", [input], [output]);
}

/**
 * Build a Split node with explicit split sizes.
 */
export function splitWithSizes(
  input: string,
  splitSizes: string,
  outputs: string[],
  axis: number = 0
): OnnxNode {
  return node("Split", [input, splitSizes], outputs, [attrInt("axis", axis)]);
}

/**
 * Build a Flip operation using Gather with reversed indices.
 * 
 * This implements torch.flip(x, [axis]) by:
 * 1. Getting the size along the axis
 * 2. Creating reversed indices [size-1, size-2, ..., 0]
 * 3. Using Gather to reorder elements
 * 
 * Note: This returns an array of nodes that must all be added to the graph.
 * The last output is 'output'.
 */
export function flipNodes(
  input: string,
  output: string,
  axis: number,
  size: number,
  uniqueNameFn: () => string
): OnnxNode[] {
  const nodes: OnnxNode[] = [];
  
  // Create reversed indices as a constant
  // For size=192, we want [191, 190, ..., 0]
  const reversedIndices = uniqueNameFn();
  
  // We can't create the initializer here, so use Range and Sub:
  // indices = size - 1 - Range(0, size)
  // But Range requires start, limit, delta as inputs...
  // 
  // Simpler approach: use Slice with step -1
  // Slice(input, starts=[0,...,size-1,...], ends=[0,...,min_int64,...], steps=[1,...,-1,...])
  
  // For axis=1, input shape [B, C, T]:
  // We want to reverse dimension 1 (C)
  // Slice with starts=[0, -1, 0], ends=[max, min, max], steps=[1, -1, 1]
  // But Slice ends is -inf when step is negative...
  // Actually for negative step, ends is exclusive going backwards
  
  // Let's use a simpler identity gather approach:
  // Create Gather(input, reversed_indices, axis=axis)
  // The indices need to be created externally as a constant
  
  // For now, just use the identity (this is a placeholder)
  // The caller should handle creating the reversed indices constant
  nodes.push(node("Gather", [input, reversedIndices], [output], [
    attrInt("axis", axis)
  ]));
  
  return nodes;
}

/**
 * Build a Flip operation placeholder.
 * 
 * IMPORTANT: The caller must also add a constant initializer named 
 * `{output}_reversed_indices` with the reversed indices [size-1, size-2, ..., 0]
 * where size is the dimension size along the axis.
 */
export function flip(
  input: string,
  indicesName: string,
  output: string,
  axis: number
): OnnxNode {
  return node("Gather", [input, indicesName], [output], [attrInt("axis", axis)]);
}

/**
 * Build a Gelu activation (ONNX 20+, or approximation).
 * Using the approximation: x * sigmoid(1.702 * x)
 */
export function gelu(input: string, output: string): OnnxNode {
  // For older ONNX versions, we'd need to implement the approximation manually
  // ONNX 20+ has native Gelu
  return node("Gelu", [input], [output]);
}

/**
 * Build a BatchNormalization node.
 */
export function batchNorm(
  input: string,
  scale: string,
  bias: string,
  mean: string,
  variance: string,
  output: string,
  epsilon: number = 1e-5,
  momentum: number = 0.9
): OnnxNode {
  return node(
    "BatchNormalization",
    [input, scale, bias, mean, variance],
    [output],
    [
      attrFloat("epsilon", epsilon),
      attrFloat("momentum", momentum),
    ]
  );
}

/**
 * Build an InstanceNormalization node.
 */
export function instanceNorm(
  input: string,
  scale: string,
  bias: string,
  output: string,
  epsilon: number = 1e-5
): OnnxNode {
  return node(
    "InstanceNormalization",
    [input, scale, bias],
    [output],
    [attrFloat("epsilon", epsilon)]
  );
}

/**
 * Build a GroupNormalization node (ONNX 18+).
 */
export function groupNorm(
  input: string,
  scale: string,
  bias: string,
  output: string,
  numGroups: number,
  epsilon: number = 1e-5
): OnnxNode {
  return node(
    "GroupNormalization",
    [input, scale, bias],
    [output],
    [
      attrInt("num_groups", numGroups),
      attrFloat("epsilon", epsilon),
    ]
  );
}
