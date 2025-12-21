/**
 * Synthesizer ONNX Graph Builder
 *
 * This module translates the RVC Synthesizer architecture into ONNX nodes.
 * It mirrors the Python Synthesizer.infer() method, building a static graph
 * that can be executed by ONNX Runtime.
 *
 * Architecture Overview
 * - TextEncoder (enc_p): Processes phone embeddings + optional pitch
 * - ResidualCouplingBlock (flow): Normalizing flow for latent transformation
 * - Generator (dec): HiFiGAN-based audio synthesis
 * - Speaker Embedding (emb_g): Conditions the model on speaker identity
 */

import {
  ParsedCheckpoint,
  OnnxGraph,
  OnnxNode,
  OnnxValueInfo,
  OnnxInitializer,
  OnnxDataType,
  TensorData,
} from "./types";
import {
  resetNameCounter,
  uniqueName,
  valueInfo,
  initializer,
  node,
  conv1d,
  convTranspose1d,
  linearNodes,
  matmul,
  layerNorm,
  leakyRelu,
  relu,
  sigmoid,
  tanh,
  add,
  mul,
  sub,
  div,
  exp,
  softmax,
  transpose,
  reshape,
  unsqueeze,
  concat,
  splitWithSizes,
  gather,
  slice,
  pad,
  sin,
  randomNormalLike,
  less,
  cast,
  range,
  where,
  flip,
  shape,
} from "./onnx-builder";

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Build sequence mask: mask[i, j] = 1 if j < lengths[i] else 0
 * 
 * Python equivalent:
 *   x = torch.arange(max_length, dtype=length.dtype, device=length.device)
 *   return x.unsqueeze(0) < length.unsqueeze(1)
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param lengths - Input tensor name containing sequence lengths [batch]
 * @param maxLength - Maximum sequence length (or tensor name)
 * @param outputName - Name for the output mask tensor
 * @param addInt64Const - Helper to add int64 constants
 * @returns Output tensor name [batch, 1, maxLength]
 */
function buildSequenceMask(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  lengths: string,
  maxLength: number | string,
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  // Create range [0, 1, 2, ..., maxLength-1]
  const zeroConst = addInt64Const(uniqueName("zero"), [0], []);
  const oneConst = addInt64Const(uniqueName("one"), [1], []);
  
  let maxLenTensor: string;
  if (typeof maxLength === "number") {
    maxLenTensor = addInt64Const(uniqueName("max_len"), [maxLength], []);
  } else {
    maxLenTensor = maxLength;
  }
  
  // range_vals = range(0, maxLength, 1) -> [maxLength]
  const rangeVals = uniqueName("range_vals");
  nodes.push(range(zeroConst, maxLenTensor, oneConst, rangeVals));
  
  // Unsqueeze range to [1, maxLength]
  const unsqueezeAxes0 = addInt64Const(uniqueName("unsqueeze_axes_0"), [0], [1]);
  const rangeUnsqueezed = uniqueName("range_unsqueezed");
  nodes.push(unsqueeze(rangeVals, unsqueezeAxes0, rangeUnsqueezed));
  
  // Unsqueeze lengths to [batch, 1]
  const unsqueezeAxes1 = addInt64Const(uniqueName("unsqueeze_axes_1"), [1], [1]);
  const lengthsUnsqueezed = uniqueName("lengths_unsqueezed");
  nodes.push(unsqueeze(lengths, unsqueezeAxes1, lengthsUnsqueezed));
  
  // Compare: range < lengths -> [batch, maxLength] bool
  const maskBool = uniqueName("mask_bool");
  nodes.push(less(rangeUnsqueezed, lengthsUnsqueezed, maskBool));
  
  // Cast bool to float32
  const maskFloat = uniqueName("mask_float");
  nodes.push(cast(maskBool, maskFloat, OnnxDataType.FLOAT));
  
  // Unsqueeze to [batch, 1, maxLength] for broadcasting with [batch, channels, length]
  const unsqueezeAxesMid = addInt64Const(uniqueName("unsqueeze_axes_mid"), [1], [1]);
  nodes.push(unsqueeze(maskFloat, unsqueezeAxesMid, outputName));
  
  return outputName;
}

/**
 * Build causal attention mask for autoregressive models.
 * 
 * Creates a lower-triangular mask where position i can only attend to positions <= i.
 * 
 * Python equivalent:
 *   mask = torch.tril(torch.ones(size, size))
 * 
 * In ONNX, we build this as:
 *   row_indices = range(size).unsqueeze(1)  # [size, 1]
 *   col_indices = range(size).unsqueeze(0)  # [1, size]
 *   mask = (row_indices >= col_indices).float()  # [size, size]
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param size - Sequence length (number or tensor name)
 * @param outputName - Name for the output mask tensor
 * @param addInt64Const - Helper to add int64 constants
/**
 * Detect weight_norm format and return the weight keys.
 * 
 * PyTorch has two weight_norm formats:
 * 1. Modern (parametrizations): `layer.parametrizations.weight.original0` (g) and `original1` (v)
 * 2. Legacy: `layer.weight_g` and `layer.weight_v`
 * 
 * @param weights - Weight map
 * @param prefix - Layer prefix (e.g., "flow.flows.0.enc.in_layers.0")
 * @returns Object with weightG and weightV keys, or null if no weight_norm
 */
function getWeightNormKeys(
  weights: Map<string, TensorData>,
  prefix: string
): { weightG: string; weightV: string } | null {
  // Remove trailing dot if present
  const p = prefix.endsWith(".") ? prefix.slice(0, -1) : prefix;
  
  // Check modern format first
  const modernG = `${p}.parametrizations.weight.original0`;
  const modernV = `${p}.parametrizations.weight.original1`;
  if (weights.has(modernG) && weights.has(modernV)) {
    return { weightG: modernG, weightV: modernV };
  }
  
  // Check legacy format
  const legacyG = `${p}.weight_g`;
  const legacyV = `${p}.weight_v`;
  if (weights.has(legacyG) && weights.has(legacyV)) {
    return { weightG: legacyG, weightV: legacyV };
  }
  
  return null;
}

/**
 * Check if a layer has weight_norm (either format).
 */
function hasWeightNorm(weights: Map<string, TensorData>, prefix: string): boolean {
  return getWeightNormKeys(weights, prefix) !== null;
}

/**
 * Pre-compute normalized weight from weight_norm decomposition at conversion time.
 * This avoids generating ReduceSum/Sqrt nodes which may cause issues in some ONNX runtimes.
 * 
 * PyTorch weight_norm stores:
 *   weight_g: magnitude [out_channels, 1, 1, ...] or [out_channels]
 *   weight_v: direction [out_channels, in_channels, kernel_size, ...]
 * 
 * Combined: weight = weight_g * (weight_v / ||weight_v||)
 *         = weight_g * weight_v / sqrt(sum(weight_v^2, dim=[1,2,...]))
 */
function precomputeNormalizedWeight(
  weightG: TensorData,
  weightV: TensorData
): Float32Array {
  const gData = weightG.data as Float32Array;
  const vData = weightV.data as Float32Array;
  const vShape = weightV.shape;
  
  const outChannels = vShape[0];
  const channelSize = vData.length / outChannels; // in_channels * kernel_size * ...
  
  const result = new Float32Array(vData.length);
  
  for (let oc = 0; oc < outChannels; oc++) {
    // Compute L2 norm of this output channel's weights
    const startIdx = oc * channelSize;
    let sumSq = 0;
    for (let i = 0; i < channelSize; i++) {
      const v = vData[startIdx + i];
      sumSq += v * v;
    }
    const norm = Math.sqrt(sumSq + 1e-12);
    
    // Get the magnitude for this channel
    // weight_g can be [out_channels] or [out_channels, 1, 1, ...]
    const g = gData[oc];
    
    // Compute normalized weight: g * v / norm
    const scale = g / norm;
    for (let i = 0; i < channelSize; i++) {
      result[startIdx + i] = vData[startIdx + i] * scale;
    }
  }
  
  return result;
}

/**
 * Reconstruct weight from weight_norm decomposition.
 * 
 * PyTorch weight_norm stores:
 *   weight_g: magnitude [out_channels, 1, 1, ...] or [out_channels]
 *   weight_v: direction [out_channels, in_channels, kernel_size, ...]
 * 
 * Combined: weight = weight_g * (weight_v / ||weight_v||)
 *         = weight_g * weight_v / sqrt(sum(weight_v^2, dim=[1,2,...]))
 * 
 * For Conv1d: weight shape is [out_channels, in_channels/groups, kernel_size]
 */
function buildWeightNormReconstruction(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  weightGKey: string,
  weightVKey: string,
  output: string
): string {
  // Get the actual tensor data
  const weightGData = weights.get(weightGKey);
  const weightVData = weights.get(weightVKey);
  
  if (!weightGData || !weightVData) {
    throw new Error(`Weight norm keys not found: ${weightGKey}, ${weightVKey}`);
  }
  
  // Pre-compute the normalized weight at conversion time
  const normalizedWeight = precomputeNormalizedWeight(weightGData, weightVData);
  
  // Add as initializer
  initializers.push(initializer(output, {
    data: normalizedWeight,
    shape: weightVData.shape,
    dtype: "float32",
  }));
  
  return output;
}

/**
 * Fused add + tanh + sigmoid + multiply operation.
 * 
 * Python equivalent (from commons.py):
 *   in_act = input_a + input_b
 *   t_act = torch.tanh(in_act[:, :n_channels, :])
 *   s_act = torch.sigmoid(in_act[:, n_channels:, :])
 *   acts = t_act * s_act
 *   return acts
 * 
 * @param nodes - Array to add nodes to
 * @param inputA - First input tensor [batch, 2*channels, time]
 * @param inputB - Second input tensor (conditioning) [batch, 2*channels, time] or scalar 0
 * @param nChannels - Number of channels (half of total)
 * @param output - Output tensor name
 * @param addInt64Const - Helper to add int64 constants
 */
function buildFusedAddTanhSigmoidMultiply(
  nodes: OnnxNode[],
  inputA: string,
  inputB: string | null,
  nChannels: number,
  output: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  // Add inputs (if inputB is provided)
  let inAct: string;
  if (inputB) {
    inAct = uniqueName("in_act");
    nodes.push(add(inputA, inputB, inAct));
  } else {
    inAct = inputA;
  }
  
  // Split along channel dimension (dim=1) into two halves
  const splitSizes = addInt64Const(uniqueName("split_sizes"), [nChannels, nChannels], [2]);
  const tanhInput = uniqueName("tanh_input");
  const sigmoidInput = uniqueName("sigmoid_input");
  nodes.push(splitWithSizes(inAct, splitSizes, [tanhInput, sigmoidInput], 1));
  
  // Apply tanh to first half
  const tAct = uniqueName("t_act");
  nodes.push(tanh(tanhInput, tAct));
  
  // Apply sigmoid to second half
  const sAct = uniqueName("s_act");
  nodes.push(sigmoid(sigmoidInput, sAct));
  
  // Multiply tanh * sigmoid
  nodes.push(mul(tAct, sAct, output));
  
  return output;
}

/**
 * Build WaveNet residual blocks (from modules.py).
 * 
 * WaveNet architecture:
 *   - cond_layer: Conv1d that projects global conditioning g to 2*hidden*n_layers
 *   - For each layer:
 *     - in_layers[i]: weight_norm Conv1d with dilation
 *     - fused_add_tanh_sigmoid_multiply(x_in, g_l)
 *     - res_skip_layers[i]: weight_norm Conv1d for residual/skip
 *     - Split res_skip into residual (to add to x) and skip (to accumulate)
 *   - Return accumulated skip output * mask
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param weights - Map of weight name -> tensor data
 * @param input - Input tensor name [batch, hidden_channels, time]
 * @param mask - Mask tensor name [batch, 1, time]
 * @param g - Global conditioning tensor name [batch, gin_channels, 1]
 * @param prefix - Weight name prefix (e.g., "flow.flows.0.")
 * @param hiddenChannels - Number of hidden channels
 * @param kernelSize - Convolution kernel size
 * @param dilationRate - Base dilation rate
 * @param nLayers - Number of WaveNet layers
 * @param addWeight - Helper to add weight initializers
 * @param addInt64Const - Helper to add int64 constants
 */
function buildWaveNet(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  input: string,
  mask: string,
  g: string | null,
  prefix: string,
  hiddenChannels: number,
  kernelSize: number,
  dilationRate: number,
  nLayers: number,
  addWeight: (name: string) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  let x = input;
  
  // Initialize output accumulator with zeros (same shape as input)
  // output = x.clone().zero_()
  const zeroConst = uniqueName("wavenet_zero");
  initializers.push(initializer(zeroConst, {
    data: new Float32Array([0.0]),
    shape: [],
    dtype: "float32",
  }));
  let output = uniqueName("wavenet_output_init");
  nodes.push(mul(input, zeroConst, output)); // Multiply by 0 to get zeros with same shape
  
  // Process global conditioning through cond_layer
  // g = self.cond_layer(g) -> [batch, 2 * hidden_channels * n_layers, 1]
  let gConditioned: string | null = null;
  const condLayerPrefix = `${prefix}cond_layer`;
  if (g && (weights.has(`${condLayerPrefix}.weight`) || hasWeightNorm(weights, condLayerPrefix))) {
    // Check for weight_norm decomposition (both modern and legacy formats)
    const wnKeys = getWeightNormKeys(weights, condLayerPrefix);
    
    let condWeight: string;
    if (wnKeys) {
      condWeight = uniqueName("cond_layer_weight");
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeys.weightG, wnKeys.weightV, condWeight);
    } else {
      condWeight = addWeight(`${condLayerPrefix}.weight`);
    }
    
    const condBias = weights.has(`${prefix}cond_layer.bias`) 
      ? addWeight(`${prefix}cond_layer.bias`) 
      : null;
    
    gConditioned = uniqueName("g_conditioned");
    nodes.push(conv1d(g, condWeight, condBias, gConditioned, 1));
  }
  
  // Process each WaveNet layer
  for (let i = 0; i < nLayers; i++) {
    const dilation = Math.pow(dilationRate, i);
    const padding = Math.floor((kernelSize * dilation - dilation) / 2);
    
    // in_layers[i] convolution with weight_norm
    const inLayerPrefix = `${prefix}in_layers.${i}`;
    const wnKeysIn = getWeightNormKeys(weights, inLayerPrefix);
    
    let inWeight: string;
    if (wnKeysIn) {
      inWeight = uniqueName(`in_layer_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysIn.weightG, wnKeysIn.weightV, inWeight);
    } else {
      inWeight = addWeight(`${inLayerPrefix}.weight`);
    }
    
    const inBias = weights.has(`${prefix}in_layers.${i}.bias`)
      ? addWeight(`${prefix}in_layers.${i}.bias`)
      : null;
    
    // x_in = in_layers[i](x) -> [batch, 2 * hidden_channels, time]
    const xIn = uniqueName(`x_in_${i}`);
    nodes.push(conv1d(x, inWeight, inBias, xIn, kernelSize, 1, padding, dilation));
    
    // Extract conditioning slice for this layer
    // g_l = g[:, i * 2 * hidden : (i + 1) * 2 * hidden, :]
    let gL: string | null = null;
    if (gConditioned) {
      const startIdx = i * 2 * hiddenChannels;
      const endIdx = (i + 1) * 2 * hiddenChannels;
      
      const startConst = addInt64Const(uniqueName(`slice_start_${i}`), [0, startIdx, 0], [3]);
      const endConst = addInt64Const(uniqueName(`slice_end_${i}`), [2147483647, endIdx, 2147483647], [3]); // INT32_MAX for safe slicing
      const axesConst = addInt64Const(uniqueName(`slice_axes_${i}`), [0, 1, 2], [3]);
      const stepsConst = addInt64Const(uniqueName(`slice_steps_${i}`), [1, 1, 1], [3]);
      
      gL = uniqueName(`g_l_${i}`);
      nodes.push(slice(gConditioned, startConst, endConst, axesConst, stepsConst, gL));
    }
    
    // Fused activation: acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels)
    const acts = uniqueName(`acts_${i}`);
    buildFusedAddTanhSigmoidMultiply(nodes, xIn, gL, hiddenChannels, acts, addInt64Const);
    
    // res_skip_layers[i] convolution with weight_norm
    // Output channels: hidden_channels if last layer, else 2 * hidden_channels
    const isLastLayer = i === nLayers - 1;
    
    const resSkipPrefix = `${prefix}res_skip_layers.${i}`;
    const wnKeysResSkip = getWeightNormKeys(weights, resSkipPrefix);
    
    let resSkipWeight: string;
    if (wnKeysResSkip) {
      resSkipWeight = uniqueName(`res_skip_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysResSkip.weightG, wnKeysResSkip.weightV, resSkipWeight);
    } else {
      resSkipWeight = addWeight(`${resSkipPrefix}.weight`);
    }
    
    const resSkipBias = weights.has(`${prefix}res_skip_layers.${i}.bias`)
      ? addWeight(`${prefix}res_skip_layers.${i}.bias`)
      : null;
    
    // res_skip_acts = res_skip_layers[i](acts)
    const resSkipActs = uniqueName(`res_skip_acts_${i}`);
    nodes.push(conv1d(acts, resSkipWeight, resSkipBias, resSkipActs, 1));
    
    if (!isLastLayer) {
      // Split res_skip_acts into residual and skip parts
      // res_acts = res_skip_acts[:, :hidden_channels, :]
      // skip_acts = res_skip_acts[:, hidden_channels:, :]
      const splitSizes = addInt64Const(uniqueName(`split_sizes_${i}`), [hiddenChannels, hiddenChannels], [2]);
      const resActs = uniqueName(`res_acts_${i}`);
      const skipActs = uniqueName(`skip_acts_${i}`);
      nodes.push(splitWithSizes(resSkipActs, splitSizes, [resActs, skipActs], 1));
      
      // x = (x + res_acts) * x_mask
      const xPlusRes = uniqueName(`x_plus_res_${i}`);
      nodes.push(add(x, resActs, xPlusRes));
      const xMasked = uniqueName(`x_masked_${i}`);
      nodes.push(mul(xPlusRes, mask, xMasked));
      x = xMasked;
      
      // output = output + skip_acts
      const newOutput = uniqueName(`output_${i}`);
      nodes.push(add(output, skipActs, newOutput));
      output = newOutput;
    } else {
      // Last layer: output = output + res_skip_acts (no split needed)
      const newOutput = uniqueName(`output_final`);
      nodes.push(add(output, resSkipActs, newOutput));
      output = newOutput;
    }
  }
  
  // Return output * x_mask
  const finalOutput = uniqueName("wavenet_final");
  nodes.push(mul(output, mask, finalOutput));
  
  return finalOutput;
}

/**
 * Build the complete Synthesizer inference graph.
 *
 * This replicates the logic of Synthesizer.infer():
 *   g = self.emb_g(sid).unsqueeze(-1)
 *   m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
 *   z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
 *   z = self.flow(z_p, x_mask, g=g, reverse=True)
 *   o = self.dec(z * x_mask, nsff0, g=g)  # if use_f0
 *   return o, x_mask, (z, z_p, m_p, logs_p)
 */
export function buildSynthesizerGraph(
  checkpoint: ParsedCheckpoint,
  phoneLen: number
): OnnxGraph {
  resetNameCounter();

  const { config, weights, useF0, version } = checkpoint;
  const hiddenDim = version === "v2" ? 768 : 256;

  const nodes: OnnxNode[] = [];
  const initializers: OnnxInitializer[] = [];
  const inputs: OnnxValueInfo[] = [];
  const outputs: OnnxValueInfo[] = [];

  // Helper to add weight as initializer
  const addWeight = (name: string): string => {
    const tensor = weights.get(name);
    if (!tensor) {
      throw new Error(`Weight not found: ${name}`);
    }
    initializers.push(initializer(name, tensor));
    return name;
  };

  // Helper to add constant tensor
  const addConstant = (
    name: string,
    data: number[] | Float32Array,
    shape: number[],
    dtype: TensorData["dtype"] = "float32"
  ): string => {
    const arr =
      data instanceof Float32Array ? data : new Float32Array(data);
    initializers.push(
      initializer(name, { data: arr, shape, dtype })
    );
    return name;
  };

  // Add scalar constant
  const addScalar = (name: string, value: number): string => {
    return addConstant(name, [value], []);
  };

  // Add int64 constant for shapes/axes
  const addInt64Const = (
    name: string,
    values: number[],
    shape: number[]
  ): string => {
    const arr = new BigInt64Array(values.map((v) => BigInt(v)));
    initializers.push(
      initializer(name, {
        data: arr,
        shape,
        dtype: "int64",
      })
    );
    return name;
  };

  // Add float scalar constant for padding values etc.
  const addFloatConst = (name: string, value: number): string => {
    const arr = new Float32Array([value]);
    initializers.push(
      initializer(name, {
        data: arr,
        shape: [],
        dtype: "float32",
      })
    );
    return name;
  };

  // ==========================================================================
  // Define Model Inputs
  // ==========================================================================

  // phone: [batch, phone_len, hidden_dim]
  inputs.push(
    valueInfo("phone", OnnxDataType.FLOAT, [1, "phone_len", hiddenDim])
  );

  // phone_lengths: [batch]
  inputs.push(valueInfo("phone_lengths", OnnxDataType.INT64, ["batch"]));

  if (useF0) {
    // pitch: [batch, phone_len]
    inputs.push(valueInfo("pitch", OnnxDataType.INT64, [1, "phone_len"]));
    // nsff0: [batch, phone_len]
    inputs.push(valueInfo("nsff0", OnnxDataType.FLOAT, [1, "phone_len"]));
  }

  // sid: [batch] - speaker ID
  inputs.push(valueInfo("sid", OnnxDataType.INT64, ["batch"]));

  // ==========================================================================
  // Speaker Embedding: g = emb_g(sid).unsqueeze(-1)
  // ==========================================================================

  const embGWeight = addWeight("emb_g.weight");

  // Gather embedding: [batch] -> [batch, gin_channels]
  const gFlat = uniqueName("g_flat");
  nodes.push(gather(embGWeight, "sid", gFlat, 0));

  // Unsqueeze to [batch, gin_channels, 1]
  const unsqueezeAxes = addInt64Const("unsqueeze_axes_neg1", [-1], [1]);
  const g = uniqueName("g");
  nodes.push(unsqueeze(gFlat, unsqueezeAxes, g));

  // ==========================================================================
  // Text Encoder: m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths)
  // ==========================================================================

  const textEncOut = buildTextEncoder(
    nodes,
    initializers,
    weights,
    config,
    useF0,
    phoneLen,
    addWeight,
    addConstant,
    addInt64Const,
    addScalar,
    addFloatConst
  );

  const { m_p, logs_p, x_mask } = textEncOut;

  // ==========================================================================
  // Sampling: z_p = (m_p + exp(logs_p) * randn * 0.66666) * x_mask
  // ==========================================================================

  // exp(logs_p)
  const expLogsP = uniqueName("exp_logs_p");
  nodes.push(exp(logs_p, expLogsP));

  // Random noise (using RandomNormalLike)
  const noise = uniqueName("noise");
  nodes.push(randomNormalLike(m_p, noise, 0.0, 1.0));

  // noise * 0.66666
  const noiseScale = addScalar("noise_scale", 0.66666);
  const scaledNoise = uniqueName("scaled_noise");
  nodes.push(mul(noise, noiseScale, scaledNoise));

  // exp(logs_p) * scaled_noise
  const expNoise = uniqueName("exp_noise");
  nodes.push(mul(expLogsP, scaledNoise, expNoise));

  // m_p + exp_noise
  const zPPreMask = uniqueName("z_p_pre_mask");
  nodes.push(add(m_p, expNoise, zPPreMask));

  // * x_mask
  const zP = uniqueName("z_p");
  nodes.push(mul(zPPreMask, x_mask, zP));

  // ==========================================================================
  // Flow (reverse): z = flow(z_p, x_mask, g=g, reverse=True)
  // ==========================================================================

  const z = buildResidualCouplingBlock(
    nodes,
    initializers,
    weights,
    config,
    zP,
    x_mask,
    g,
    true, // reverse
    addWeight,
    addConstant,
    addInt64Const,
    addScalar
  );

  // z * x_mask for decoder input
  const zMasked = uniqueName("z_masked");
  nodes.push(mul(z, x_mask, zMasked));

  // ==========================================================================
  // Decoder (HiFiGAN): o = dec(z * x_mask, nsff0, g=g)
  // ==========================================================================

  const audio = buildHiFiGANDecoder(
    nodes,
    initializers,
    weights,
    config,
    zMasked,
    useF0 ? "nsff0" : null,
    g,
    useF0,
    addWeight,
    addConstant,
    addInt64Const,
    addScalar
  );

  // ==========================================================================
  // Outputs
  // ==========================================================================

  // audio: [batch, 1, audio_len] - use symbolic batch like Python
  outputs.push(valueInfo("audio", OnnxDataType.FLOAT, ["batch", 1, "audio_len"]));

  // Rename final audio output
  nodes.push(node("Identity", [audio], ["audio"]));

  // Add sample rate as constant output
  const srConst = addInt64Const("sr_const", [config.sr], [1]);
  nodes.push(node("Identity", [srConst], ["sr"]));
  outputs.push(valueInfo("sr", OnnxDataType.INT64, [1]));

  return {
    name: "RVC_Synthesizer",
    nodes,
    inputs,
    outputs,
    initializers,
  };
}

// =============================================================================
// Text Encoder Builder
// =============================================================================

interface TextEncoderOutput {
  m_p: string;
  logs_p: string;
  x_mask: string;
}

function buildTextEncoder(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  useF0: boolean,
  phoneLen: number,  // Added: sequence length for relative positional encoding
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string,
  addFloatConst: (name: string, value: number) => string
): TextEncoderOutput {
  const prefix = "enc_p.";

  // Linear embedding: emb_phone(phone)
  const embPhoneWeight = addWeight(`${prefix}emb_phone.weight`);
  const embPhoneBias = addWeight(`${prefix}emb_phone.bias`);

  // phone: [B, T, H_in] -> [B, T, H_out]
  const phoneEmbedded = uniqueName("phone_embedded");
  nodes.push(...linearNodes("phone", embPhoneWeight, embPhoneBias, phoneEmbedded));

  let x = phoneEmbedded;

  // Pitch embedding (if useF0)
  if (useF0 && weights.has(`${prefix}emb_pitch.weight`)) {
    const embPitchWeight = addWeight(`${prefix}emb_pitch.weight`);
    const pitchEmbedded = uniqueName("pitch_embedded");
    nodes.push(gather(embPitchWeight, "pitch", pitchEmbedded, 0));

    // Add pitch embedding to phone embedding
    const xWithPitch = uniqueName("x_with_pitch");
    nodes.push(add(x, pitchEmbedded, xWithPitch));
    x = xWithPitch;
  }

  // Scale by sqrt(hidden_channels)
  const scale = addScalar("enc_scale", Math.sqrt(config.hiddenChannels));
  const xScaled = uniqueName("x_scaled");
  nodes.push(mul(x, scale, xScaled));

  // LeakyReLU
  const xActivated = uniqueName("x_activated");
  nodes.push(leakyRelu(xScaled, xActivated, 0.1));

  // Transpose: [B, T, H] -> [B, H, T]
  const xTransposed = uniqueName("x_transposed");
  nodes.push(transpose(xActivated, xTransposed, [0, 2, 1]));

  // Create sequence mask from phone_lengths
  // mask[i, j] = 1.0 if j < phone_lengths[i] else 0.0
  // Shape: [batch, 1, phone_len]
  
  // First, get the max length from phone input shape
  // phone shape is [batch, phone_len, hidden] so we need dim 1
  const phoneShape = uniqueName("phone_shape");
  nodes.push(shape("phone", phoneShape));
  
  // Extract phone_len (index 1)
  const dimOneIdx = addInt64Const("dim_one_idx", [1], [1]);
  const phoneLenDynamic = uniqueName("phone_len_dynamic");
  nodes.push(gather(phoneShape, dimOneIdx, phoneLenDynamic, 0));
  
  // Build proper sequence mask using Range + Less + Cast
  const x_mask = uniqueName("x_mask");
  buildSequenceMask(nodes, initializers, "phone_lengths", phoneLenDynamic, x_mask, addInt64Const);

  // Apply mask
  let xMasked = uniqueName("x_masked");
  nodes.push(mul(xTransposed, x_mask, xMasked));

  // Transformer Encoder layers
  xMasked = buildTransformerEncoder(
    nodes,
    initializers,
    weights,
    config,
    xMasked,
    x_mask,
    `${prefix}encoder.`,
    phoneLen,
    addWeight,
    addConstant,
    addInt64Const,
    addScalar,
    addFloatConst
  );

  // Projection: proj(x) -> [B, out_channels * 2, T]
  const projWeight = addWeight(`${prefix}proj.weight`);
  const projBias = addWeight(`${prefix}proj.bias`);
  const stats = uniqueName("stats");
  nodes.push(conv1d(xMasked, projWeight, projBias, stats, 1));

  // Mask stats
  const statsMasked = uniqueName("stats_masked");
  nodes.push(mul(stats, x_mask, statsMasked));

  // Split into m_p and logs_p
  // Projection outputs inter_channels * 2, split into two equal halves
  const splitSizes = addInt64Const(uniqueName("stats_split_sizes"), 
    [config.interChannels, config.interChannels], [2]);
  const m_p = uniqueName("m_p");
  const logs_p = uniqueName("logs_p");
  nodes.push(splitWithSizes(statsMasked, splitSizes, [m_p, logs_p], 1));

  return { m_p, logs_p, x_mask };
}

// =============================================================================
// Transformer Encoder Builder
// =============================================================================

function buildTransformerEncoder(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  mask: string,
  prefix: string,
  phoneLen: number,  // Added: sequence length for relative positional encoding
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  let x = input;
  
  // Python: attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
  // Creates [B, 1, T, T] 2D attention mask where position [i,j] is 1 only if both are valid
  // mask is [B, 1, T], need [B, 1, T, 1] * [B, 1, 1, T] = [B, 1, T, T]
  const maskUnsqueeze2Axes = addInt64Const(uniqueName("mask_unsqueeze2_axes"), [3], [1]);
  const maskUnsqueeze2 = uniqueName("mask_unsqueeze2");
  nodes.push(unsqueeze(mask, maskUnsqueeze2Axes, maskUnsqueeze2));  // [B, 1, T, 1]
  
  const maskUnsqueezeMinus1Axes = addInt64Const(uniqueName("mask_unsqueeze_m1_axes"), [2], [1]);
  const maskUnsqueezeMinus1 = uniqueName("mask_unsqueeze_minus1");
  nodes.push(unsqueeze(mask, maskUnsqueezeMinus1Axes, maskUnsqueezeMinus1));  // [B, 1, 1, T]
  
  const attnMask = uniqueName("attn_mask");
  nodes.push(mul(maskUnsqueeze2, maskUnsqueezeMinus1, attnMask));  // [B, 1, T, T]

  for (let i = 0; i < config.nLayers; i++) {
    const layerPrefix = `${prefix}`;

    // Self-Attention
    const attnOut = buildMultiHeadAttention(
      nodes,
      initializers,
      weights,
      config,
      x,
      x,
      attnMask,  // Use 2D attention mask
      `${layerPrefix}attn_layers.${i}.`,
      phoneLen,
      addWeight,
      addConstant,
      addInt64Const,
      addScalar,
      addFloatConst
    );

    // Residual + LayerNorm 1
    const xResidual1 = uniqueName("x_residual1");
    nodes.push(add(x, attnOut, xResidual1));

    const normWeight1 = addWeight(`${layerPrefix}norm_layers_1.${i}.gamma`);
    const normBias1 = addWeight(`${layerPrefix}norm_layers_1.${i}.beta`);
    const xNorm1 = uniqueName("x_norm1");
    // LayerNorm on last dim (channels), but input is [B, C, T]
    // Transpose -> LayerNorm -> Transpose back
    const xT1 = uniqueName("x_transpose1");
    nodes.push(transpose(xResidual1, xT1, [0, 2, 1]));
    const xLN1 = uniqueName("x_ln1");
    nodes.push(layerNorm(xT1, normWeight1, normBias1, xLN1, -1, 1e-5));
    nodes.push(transpose(xLN1, xNorm1, [0, 2, 1]));

    // FFN
    const ffnOut = buildFFN(
      nodes,
      initializers,
      weights,
      config,
      xNorm1,
      mask,
      `${layerPrefix}ffn_layers.${i}.`,
      addWeight,
      addConstant,
      addInt64Const,
      addScalar
    );

    // Residual + LayerNorm 2
    const xResidual2 = uniqueName("x_residual2");
    nodes.push(add(xNorm1, ffnOut, xResidual2));

    const normWeight2 = addWeight(`${layerPrefix}norm_layers_2.${i}.gamma`);
    const normBias2 = addWeight(`${layerPrefix}norm_layers_2.${i}.beta`);
    const xT2 = uniqueName("x_transpose2");
    nodes.push(transpose(xResidual2, xT2, [0, 2, 1]));
    const xLN2 = uniqueName("x_ln2");
    nodes.push(layerNorm(xT2, normWeight2, normBias2, xLN2, -1, 1e-5));
    const xNorm2 = uniqueName("x_norm2");
    nodes.push(transpose(xLN2, xNorm2, [0, 2, 1]));

    x = xNorm2;
  }

  // Final mask application
  const xFinal = uniqueName("x_enc_final");
  nodes.push(mul(x, mask, xFinal));

  return xFinal;
}

// =============================================================================
// Relative Positional Encoding Helpers
// =============================================================================

/**
 * Build relative embeddings extraction for a given sequence length.
 * 
 * Python equivalent (from attentions.py):
 *   def _get_relative_embeddings(self, embeddings, length):
 *       pad_length = max(length - (self.window_size + 1), 0)
 *       start = max((self.window_size + 1) - length, 0)
 *       end = start + 2 * length - 1
 *       if pad_length > 0:
 *           embeddings = F.pad(embeddings, [0, 0, pad_length, pad_length, 0, 0])
 *       return embeddings[:, start:end]
 * 
 * Uses dynamic shapes for variable-length support.
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param embeddings - Embedding tensor name [1 or n_heads, 2*window_size+1, k_channels]
 * @param windowSize - Window size for relative positions (static, from model config)
 * @param timeDim - Dynamic length tensor from runtime shape
 * @param outputName - Name for the output tensor
 * @param addInt64Const - Helper to add int64 constants
 * @returns Output tensor name [1 or n_heads, 2*length-1, k_channels]
 */
function buildGetRelativeEmbeddings(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  embeddings: string,
  windowSize: number,
  timeDim: string, // Dynamic tensor representing sequence length
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  // Compute derived values dynamically
  const oneConst = addInt64Const(uniqueName("rel_one"), [1], [1]);
  const twoConst = addInt64Const(uniqueName("rel_two"), [2], [1]);
  const zeroConst = addInt64Const(uniqueName("rel_zero"), [0], [1]);
  const windowPlus1 = addInt64Const(uniqueName("rel_window_plus_1"), [windowSize + 1], [1]);
  
  // padLength = max(length - (windowSize + 1), 0)
  const lengthMinusWindowPlus1 = uniqueName("rel_len_minus_wp1");
  nodes.push(sub(timeDim, windowPlus1, lengthMinusWindowPlus1));
  const padLength = uniqueName("rel_pad_length");
  nodes.push(node("Max", [lengthMinusWindowPlus1, zeroConst], [padLength]));
  
  // start = max((windowSize + 1) - length, 0)
  const windowPlus1MinusLength = uniqueName("rel_wp1_minus_len");
  nodes.push(sub(windowPlus1, timeDim, windowPlus1MinusLength));
  const start = uniqueName("rel_start");
  nodes.push(node("Max", [windowPlus1MinusLength, zeroConst], [start]));
  
  // end = start + 2 * length - 1
  const twoTimesLength = uniqueName("rel_two_times_len");
  nodes.push(mul(twoConst, timeDim, twoTimesLength));
  const startPlusTwoLen = uniqueName("rel_start_plus_two_len");
  nodes.push(add(start, twoTimesLength, startPlusTwoLen));
  const end = uniqueName("rel_end");
  nodes.push(sub(startPlusTwoLen, oneConst, end));
  
  // Pad: [0, 0] for k_channels, [padLength, padLength] for positions, [0, 0] for heads
  // ONNX Pad format for 3D: [dim0_before, dim1_before, dim2_before, dim0_after, dim1_after, dim2_after]
  const padsShape = uniqueName("rel_pads_shape");
  nodes.push(concat([zeroConst, padLength, zeroConst, zeroConst, padLength, zeroConst], padsShape, 0));
  const padValue = addFloatConst(uniqueName("rel_pad_value"), 0.0);
  const paddedEmb = uniqueName("rel_emb_padded");
  nodes.push(pad(embeddings, padsShape, paddedEmb, "constant", padValue));
  
  // Slice: embeddings[:, start:end, :]
  const axesConst = addInt64Const(uniqueName("rel_axes"), [1], [1]);
  const stepsConst = addInt64Const(uniqueName("rel_steps"), [1], [1]);
  nodes.push(slice(paddedEmb, start, end, axesConst, stepsConst, outputName));
  
  return outputName;
}

/**
 * Build matmul with relative keys: torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
 * 
 * x: [batch, heads, length, k_channels]
 * y: [1, 2*length-1, k_channels]
 * result: [batch, heads, length, 2*length-1]
 */
function buildMatmulWithRelativeKeys(
  nodes: OnnxNode[],
  query: string,  // [batch, heads, length, k_channels]
  relEmb: string, // [1, 2*length-1, k_channels]
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  // relEmb: [1, 2*length-1, k_channels]
  // unsqueeze(0): [1, 1, 2*length-1, k_channels]
  const unsqueezeAxes = addInt64Const(uniqueName("rel_unsqueeze_axes"), [0], [1]);
  const relEmbUnsqueezed = uniqueName("rel_emb_unsqueezed");
  nodes.push(unsqueeze(relEmb, unsqueezeAxes, relEmbUnsqueezed));
  
  // transpose(-2, -1): [1, 1, k_channels, 2*length-1]
  const relEmbT = uniqueName("rel_emb_transposed");
  nodes.push(transpose(relEmbUnsqueezed, relEmbT, [0, 1, 3, 2]));
  
  // matmul: [batch, heads, length, k_channels] @ [1, 1, k_channels, 2*length-1]
  // broadcasts to: [batch, heads, length, 2*length-1]
  nodes.push(matmul(query, relEmbT, outputName));
  
  return outputName;
}

/**
 * Build matmul with relative values: torch.matmul(x, y.unsqueeze(0))
 * 
 * x: [batch, heads, length, 2*length-1]
 * y: [1, 2*length-1, k_channels]
 * result: [batch, heads, length, k_channels]
 */
function buildMatmulWithRelativeValues(
  nodes: OnnxNode[],
  relWeights: string,  // [batch, heads, length, 2*length-1]
  relEmb: string,      // [1, 2*length-1, k_channels]
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  // relEmb: [1, 2*length-1, k_channels]
  // unsqueeze(0): [1, 1, 2*length-1, k_channels]
  const unsqueezeAxes = addInt64Const(uniqueName("relv_unsqueeze_axes"), [0], [1]);
  const relEmbUnsqueezed = uniqueName("relv_emb_unsqueezed");
  nodes.push(unsqueeze(relEmb, unsqueezeAxes, relEmbUnsqueezed));
  
  // matmul: [batch, heads, length, 2*length-1] @ [1, 1, 2*length-1, k_channels]
  // broadcasts to: [batch, heads, length, k_channels]
  nodes.push(matmul(relWeights, relEmbUnsqueezed, outputName));
  
  return outputName;
}

/**
 * Build relative position to absolute position conversion.
 * 
 * Converts relative position logits to absolute position scores.
 * Uses dynamic shapes extracted from input tensor for variable-length support.
 * 
 * Python equivalent:
 *   def _relative_position_to_absolute_position(self, x):
 *       batch, heads, length, _ = x.size()  # _ is 2*length-1
 *       x = F.pad(x, [0, 1])  # pad last dim
 *       x_flat = x.view(batch, heads, length * 2 * length)
 *       x_flat = F.pad(x_flat, [0, length - 1])
 *       return x_flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]
 * 
 * Input: [batch, heads, length, 2*length-1]
 * Output: [batch, heads, length, length]
 */
function buildRelativeToAbsolutePosition(
  nodes: OnnxNode[],
  relLogits: string,
  batchDim: string,
  headsDim: string,
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  // Extract length dynamically from input shape [batch, heads, length, 2*length-1]
  const inputShape = uniqueName("r2a_input_shape");
  nodes.push(shape(relLogits, inputShape));
  
  const dim2Idx = addInt64Const(uniqueName("r2a_dim2_idx"), [2], [1]);
  const lengthDim = uniqueName("r2a_length"); // This is the sequence length as a scalar tensor
  nodes.push(gather(inputShape, dim2Idx, lengthDim, 0));
  
  // Compute derived values dynamically
  const oneConst = addInt64Const(uniqueName("r2a_one"), [1], [1]);
  const twoConst = addInt64Const(uniqueName("r2a_two"), [2], [1]);
  
  // length + 1
  const lengthPlus1 = uniqueName("r2a_len_plus_1");
  nodes.push(add(lengthDim, oneConst, lengthPlus1));
  
  // length - 1
  const lengthMinus1 = uniqueName("r2a_len_minus_1");
  nodes.push(sub(lengthDim, oneConst, lengthMinus1));
  
  // 2 * length
  const twoLen = uniqueName("r2a_two_len");
  nodes.push(mul(twoConst, lengthDim, twoLen));
  
  // 2 * length - 1
  const twoLenMinus1 = uniqueName("r2a_two_len_minus_1");
  nodes.push(sub(twoLen, oneConst, twoLenMinus1));
  
  // length * 2 * length = length * twoLen
  const flatSize = uniqueName("r2a_flat_size");
  nodes.push(mul(lengthDim, twoLen, flatSize));
  
  // Pad last dimension with 1 zero: [0, 1]
  const pad1Const = addInt64Const(uniqueName("r2a_pad1"), [0, 0, 0, 0, 0, 0, 0, 1], [8]);
  const padValue1 = addFloatConst(uniqueName("r2a_pad_val1"), 0.0);
  const padded1 = uniqueName("r2a_padded1");
  nodes.push(pad(relLogits, pad1Const, padded1, "constant", padValue1));
  // Shape now: [batch, heads, length, 2*length]
  
  // Flatten: view(batch, heads, length * 2 * length)
  const flatShape = uniqueName("r2a_flat_shape");
  nodes.push(concat([batchDim, headsDim, flatSize], flatShape, 0));
  const flattened = uniqueName("r2a_flattened");
  nodes.push(reshape(padded1, flatShape, flattened));
  // Shape now: [batch, heads, length * 2 * length]
  
  // Pad: [0, length - 1] - need to build dynamic padding
  // Pad expects [begin_0, end_0, begin_1, end_1, ...] for each dimension
  // For 3D tensor, we need [0, 0, 0, 0, 0, length-1]
  const zeroConst = addInt64Const(uniqueName("r2a_zero"), [0], [1]);
  const pad2Shape = uniqueName("r2a_pad2_shape");
  nodes.push(concat([zeroConst, zeroConst, zeroConst, zeroConst, zeroConst, lengthMinus1], pad2Shape, 0));
  const padValue2 = addFloatConst(uniqueName("r2a_pad_val2"), 0.0);
  const padded2 = uniqueName("r2a_padded2");
  nodes.push(pad(flattened, pad2Shape, padded2, "constant", padValue2));
  // Shape now: [batch, heads, length * 2 * length + length - 1]
  
  // Reshape to [batch, heads, length + 1, 2 * length - 1]
  const viewShape = uniqueName("r2a_view_shape");
  nodes.push(concat([batchDim, headsDim, lengthPlus1, twoLenMinus1], viewShape, 0));
  const reshaped = uniqueName("r2a_reshaped");
  nodes.push(reshape(padded2, viewShape, reshaped));
  // Shape now: [batch, heads, length + 1, 2 * length - 1]
  
  // Slice: [:, :, :length, length-1:]
  // First slice dim 2: [:length]
  const startDim2 = addInt64Const(uniqueName("r2a_start_dim2"), [0], [1]);
  const axesDim2 = addInt64Const(uniqueName("r2a_axes_dim2"), [2], [1]);
  const stepsDim2 = addInt64Const(uniqueName("r2a_steps_dim2"), [1], [1]);
  const sliced1 = uniqueName("r2a_sliced1");
  nodes.push(slice(reshaped, startDim2, lengthDim, axesDim2, stepsDim2, sliced1));
  // Shape now: [batch, heads, length, 2 * length - 1]
  
  // Then slice dim 3: [length-1:]
  const axesDim3 = addInt64Const(uniqueName("r2a_axes_dim3"), [3], [1]);
  const stepsDim3 = addInt64Const(uniqueName("r2a_steps_dim3"), [1], [1]);
  nodes.push(slice(sliced1, lengthMinus1, twoLenMinus1, axesDim3, stepsDim3, outputName));
  // Shape now: [batch, heads, length, length]
  
  return outputName;
}

/**
 * Build absolute position to relative position conversion.
 * 
 * Converts absolute attention weights to relative weights for value lookup.
 * Uses dynamic shapes extracted from input tensor for variable-length support.
 * 
 * Python equivalent:
 *   def _absolute_position_to_relative_position(self, x):
 *       batch, heads, length, _ = x.size()  # _ is length
 *       x = F.pad(x, [0, length - 1])  # pad last dim
 *       x_flat = x.view(batch, heads, length**2 + length*(length-1))
 *       x_flat = F.pad(x_flat, [length, 0])  # pad beginning
 *       return x_flat.view(batch, heads, length, 2*length)[:, :, :, 1:]
 * 
 * Input: [batch, heads, length, length]
 * Output: [batch, heads, length, 2*length-1]
 */
function buildAbsoluteToRelativePosition(
  nodes: OnnxNode[],
  absWeights: string,
  batchDim: string,
  headsDim: string,
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  // Extract length dynamically from input shape [batch, heads, length, length]
  const inputShape = uniqueName("a2r_input_shape");
  nodes.push(shape(absWeights, inputShape));
  
  const dim2Idx = addInt64Const(uniqueName("a2r_dim2_idx"), [2], [1]);
  const lengthDim = uniqueName("a2r_length"); // This is the sequence length as a scalar tensor
  nodes.push(gather(inputShape, dim2Idx, lengthDim, 0));
  
  // Compute derived values dynamically
  const oneConst = addInt64Const(uniqueName("a2r_one"), [1], [1]);
  const twoConst = addInt64Const(uniqueName("a2r_two"), [2], [1]);
  const zeroConst = addInt64Const(uniqueName("a2r_zero"), [0], [1]);
  
  // length - 1
  const lengthMinus1 = uniqueName("a2r_len_minus_1");
  nodes.push(sub(lengthDim, oneConst, lengthMinus1));
  
  // 2 * length
  const twoLen = uniqueName("a2r_two_len");
  nodes.push(mul(twoConst, lengthDim, twoLen));
  
  // length^2 = length * length
  const lengthSquared = uniqueName("a2r_len_squared");
  nodes.push(mul(lengthDim, lengthDim, lengthSquared));
  
  // length * (length - 1)
  const lenTimesLenMinus1 = uniqueName("a2r_len_times_len_minus_1");
  nodes.push(mul(lengthDim, lengthMinus1, lenTimesLenMinus1));
  
  // flatSize = length^2 + length*(length-1)
  const flatSize = uniqueName("a2r_flat_size");
  nodes.push(add(lengthSquared, lenTimesLenMinus1, flatSize));
  
  // Pad last dimension: [0, length - 1] - build dynamically
  // ONNX pads format for 4D: [dim0_b, dim1_b, dim2_b, dim3_b, dim0_a, dim1_a, dim2_a, dim3_a]
  const pad1Shape = uniqueName("a2r_pad1_shape");
  nodes.push(concat([zeroConst, zeroConst, zeroConst, zeroConst, zeroConst, zeroConst, zeroConst, lengthMinus1], pad1Shape, 0));
  const padValue1 = addFloatConst(uniqueName("a2r_pad_val1"), 0.0);
  const padded1 = uniqueName("a2r_padded1");
  nodes.push(pad(absWeights, pad1Shape, padded1, "constant", padValue1));
  // Shape now: [batch, heads, length, 2*length - 1]
  
  // Flatten: view(batch, heads, flatSize)
  const flatShape = uniqueName("a2r_flat_shape");
  nodes.push(concat([batchDim, headsDim, flatSize], flatShape, 0));
  const flattened = uniqueName("a2r_flattened");
  nodes.push(reshape(padded1, flatShape, flattened));
  // Shape now: [batch, heads, flatSize]
  
  // Pad beginning of last dimension: [length, 0] - build dynamically
  // ONNX pads format for 3D: [dim0_before, dim1_before, dim2_before, dim0_after, dim1_after, dim2_after]
  const pad2Shape = uniqueName("a2r_pad2_shape");
  nodes.push(concat([zeroConst, zeroConst, lengthDim, zeroConst, zeroConst, zeroConst], pad2Shape, 0));
  const padValue2 = addFloatConst(uniqueName("a2r_pad_val2"), 0.0);
  const padded2 = uniqueName("a2r_padded2");
  nodes.push(pad(flattened, pad2Shape, padded2, "constant", padValue2));
  // Shape now: [batch, heads, flatSize + length]
  
  // Reshape to [batch, heads, length, 2 * length]
  const viewShape = uniqueName("a2r_view_shape");
  nodes.push(concat([batchDim, headsDim, lengthDim, twoLen], viewShape, 0));
  const reshaped = uniqueName("a2r_reshaped");
  nodes.push(reshape(padded2, viewShape, reshaped));
  // Shape now: [batch, heads, length, 2 * length]
  
  // Slice: [:, :, :, 1:]
  const startDim3 = addInt64Const(uniqueName("a2r_start"), [1], [1]);
  const axesDim3 = addInt64Const(uniqueName("a2r_axes"), [3], [1]);
  const stepsDim3 = addInt64Const(uniqueName("a2r_steps"), [1], [1]);
  nodes.push(slice(reshaped, startDim3, twoLen, axesDim3, stepsDim3, outputName));
  // Shape now: [batch, heads, length, 2*length - 1]
  
  return outputName;
}

/**
 * Compute relative position scores to add to attention scores.
 * 
 * Python equivalent:
 *   def _compute_relative_scores(self, query, length):
 *       rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
 *       rel_logits = self._matmul_with_relative_keys(query / sqrt(k_channels), rel_emb)
 *       return self._relative_position_to_absolute_position(rel_logits)
 * 
 * @param query - Already scaled query [batch, heads, length, k_channels]
 * @param embRelK - Relative key embeddings [1, 2*window_size+1, k_channels]
 * @returns Relative scores [batch, heads, length, length]
 */
function buildComputeRelativeScores(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  queryScaled: string,
  embRelK: string,
  batchDim: string,
  headsDim: string,
  windowSize: number,
  timeDim: string, // Dynamic tensor representing sequence length
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  // Get relative embeddings for this length
  const relEmbK = uniqueName("rel_emb_k");
  buildGetRelativeEmbeddings(nodes, initializers, embRelK, windowSize, timeDim, relEmbK, addInt64Const, addFloatConst);
  
  // Compute matmul with relative keys
  const relLogits = uniqueName("rel_logits");
  buildMatmulWithRelativeKeys(nodes, queryScaled, relEmbK, relLogits, addInt64Const);
  
  // Convert to absolute positions (uses dynamic shapes internally)
  buildRelativeToAbsolutePosition(nodes, relLogits, batchDim, headsDim, outputName, addInt64Const, addFloatConst);
  
  return outputName;
}

/**
 * Apply relative values to attention output.
 * 
 * Python equivalent:
 *   def _apply_relative_values(self, p_attn, length):
 *       rel_weights = self._absolute_position_to_relative_position(p_attn)
 *       rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
 *       return self._matmul_with_relative_values(rel_weights, rel_emb)
 * 
 * @param pAttn - Attention weights [batch, heads, length, length]
 * @param embRelV - Relative value embeddings [1, 2*window_size+1, k_channels]
 * @returns Relative values to add to output [batch, heads, length, k_channels]
 */
function buildApplyRelativeValues(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  pAttn: string,
  embRelV: string,
  batchDim: string,
  headsDim: string,
  windowSize: number,
  timeDim: string, // Dynamic tensor representing sequence length
  outputName: string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  // Convert absolute attention to relative weights (uses dynamic shapes internally)
  const relWeights = uniqueName("rel_weights");
  buildAbsoluteToRelativePosition(nodes, pAttn, batchDim, headsDim, relWeights, addInt64Const, addFloatConst);
  
  // Get relative embeddings for this length
  const relEmbV = uniqueName("rel_emb_v");
  buildGetRelativeEmbeddings(nodes, initializers, embRelV, windowSize, timeDim, relEmbV, addInt64Const, addFloatConst);
  
  // Compute matmul with relative values
  buildMatmulWithRelativeValues(nodes, relWeights, relEmbV, outputName, addInt64Const);
  
  return outputName;
}

// =============================================================================
// Multi-Head Attention Builder
// =============================================================================

/**
 * Build Multi-Head Attention with proper reshape and optional relative positional encoding.
 * 
 * Python equivalent (from attentions.py):
 *   q, k, v = conv_q(x), conv_k(c), conv_v(c)
 *   # Reshape to [batch, n_heads, k_channels, time] then transpose to [batch, n_heads, time, k_channels]
 *   query = query.view(b, n_heads, k_channels, t_t).transpose(2, 3)
 *   key = key.view(b, n_heads, k_channels, t_s).transpose(2, 3)
 *   value = value.view(b, n_heads, k_channels, t_s).transpose(2, 3)
 *   scores = torch.matmul(query / sqrt(k_channels), key.transpose(-2, -1))
 *   # Optional: add relative positional scores
 *   if window_size: scores += relative_scores(query, t_s)
 *   # Apply mask
 *   scores = scores.masked_fill(mask == 0, -1e4)
 *   p_attn = softmax(scores, dim=-1)
 *   output = torch.matmul(p_attn, value)
 *   # Optional: add relative values
 *   output = output.transpose(2, 3).view(b, d, t_t)
 *   return conv_o(output)
 */
function buildMultiHeadAttention(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  query: string,
  key: string,
  mask: string,
  prefix: string,
  phoneLen: number,  // Added: sequence length for relative position calculation
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string,
  addFloatConst: (name: string, value: number) => string
): string {
  const nHeads = config.nHeads;
  const channels = config.hiddenChannels;
  const kChannels = channels / nHeads;
  
  // Check for relative positional encoding weights
  const hasRelativePos = weights.has(`${prefix}emb_rel_k`);
  let embRelK: string | null = null;
  let embRelV: string | null = null;
  let windowSize = 0;
  
  if (hasRelativePos) {
    embRelK = addWeight(`${prefix}emb_rel_k`);
    embRelV = addWeight(`${prefix}emb_rel_v`);
    // Infer window_size from emb_rel_k shape: [n_heads_rel, 2*window_size+1, k_channels]
    const embShape = weights.get(`${prefix}emb_rel_k`)!.shape;
    windowSize = Math.floor((embShape[1] - 1) / 2);
  }
  
  // Q, K, V projections using Conv1d
  
  // Q, K, V projections using Conv1d
  const convQWeight = addWeight(`${prefix}conv_q.weight`);
  const convQBias = addWeight(`${prefix}conv_q.bias`);
  const convKWeight = addWeight(`${prefix}conv_k.weight`);
  const convKBias = addWeight(`${prefix}conv_k.bias`);
  const convVWeight = addWeight(`${prefix}conv_v.weight`);
  const convVBias = addWeight(`${prefix}conv_v.bias`);
  const convOWeight = addWeight(`${prefix}conv_o.weight`);
  const convOBias = addWeight(`${prefix}conv_o.bias`);

  // Project Q, K, V: [batch, channels, time] -> [batch, channels, time]
  const qProj = uniqueName("q_proj");
  const kProj = uniqueName("k_proj");
  const vProj = uniqueName("v_proj");
  nodes.push(conv1d(query, convQWeight, convQBias, qProj, 1));
  nodes.push(conv1d(key, convKWeight, convKBias, kProj, 1));
  nodes.push(conv1d(key, convVWeight, convVBias, vProj, 1));

  // Get time dimension from query shape for reshape
  const qShape = uniqueName("q_shape");
  nodes.push(shape(qProj, qShape));
  
  // Extract batch and time dimensions
  const batchIdx = addInt64Const(uniqueName("batch_idx"), [0], [1]);
  const timeIdx = addInt64Const(uniqueName("time_idx"), [2], [1]);
  const batchDim = uniqueName("batch_dim");
  const timeDim = uniqueName("time_dim");
  nodes.push(gather(qShape, batchIdx, batchDim, 0));
  nodes.push(gather(qShape, timeIdx, timeDim, 0));

  // Reshape Q, K, V: [batch, channels, time] -> [batch, n_heads, k_channels, time]
  // Then transpose to [batch, n_heads, time, k_channels]
  const nHeadsConst = addInt64Const(uniqueName("n_heads"), [nHeads], [1]);
  const kChannelsConst = addInt64Const(uniqueName("k_channels"), [kChannels], [1]);
  
  // Build reshape target: [batch, n_heads, k_channels, time]
  const reshapeShape = uniqueName("reshape_shape");
  nodes.push(concat([batchDim, nHeadsConst, kChannelsConst, timeDim], reshapeShape, 0));
  
  // Reshape and transpose Q
  const qReshaped = uniqueName("q_reshaped");
  nodes.push(reshape(qProj, reshapeShape, qReshaped));
  const qHeads = uniqueName("q_heads");
  nodes.push(transpose(qReshaped, qHeads, [0, 1, 3, 2])); // [batch, n_heads, time, k_channels]
  
  // Reshape and transpose K
  const kReshaped = uniqueName("k_reshaped");
  nodes.push(reshape(kProj, reshapeShape, kReshaped));
  const kHeads = uniqueName("k_heads");
  nodes.push(transpose(kReshaped, kHeads, [0, 1, 3, 2])); // [batch, n_heads, time, k_channels]
  
  // Reshape and transpose V
  const vReshaped = uniqueName("v_reshaped");
  nodes.push(reshape(vProj, reshapeShape, vReshaped));
  const vHeads = uniqueName("v_heads");
  nodes.push(transpose(vReshaped, vHeads, [0, 1, 3, 2])); // [batch, n_heads, time, k_channels]

  // Scale query by 1/sqrt(k_channels) before matmul for numerical stability
  const scaleFactor = addScalar(uniqueName("attn_scale"), 1.0 / Math.sqrt(kChannels));
  const qScaled = uniqueName("q_scaled");
  nodes.push(mul(qHeads, scaleFactor, qScaled));
  
  // Transpose K for matmul: [batch, n_heads, time, k_channels] -> [batch, n_heads, k_channels, time]
  const kT = uniqueName("k_transposed");
  nodes.push(transpose(kHeads, kT, [0, 1, 3, 2]));
  
  // Attention scores: Q_scaled @ K^T -> [batch, n_heads, time_q, time_k]
  let scoresWithRelative = uniqueName("attn_scores");
  nodes.push(matmul(qScaled, kT, scoresWithRelative));

  // Add relative positional scores if available
  // Python: if self.window_size: scores += self._compute_relative_scores(query, t_s)
  if (hasRelativePos && embRelK) {
    // We need batch and heads dimensions for the relative position functions
    const relBatchDim = uniqueName("rel_batch_dim");
    const relHeadsDim = uniqueName("rel_heads_dim");
    const headsIdx = addInt64Const(uniqueName("heads_idx"), [1], [1]);
    nodes.push(gather(qShape, batchIdx, relBatchDim, 0));
    nodes.push(gather(qShape, headsIdx, relHeadsDim, 0));
    
    const relScores = uniqueName("rel_scores");
    buildComputeRelativeScores(
      nodes, initializers, qScaled, embRelK,
      relBatchDim, nHeadsConst, windowSize, timeDim,
      relScores, addInt64Const, addFloatConst
    );
    
    const scoresWithRel = uniqueName("scores_with_rel");
    nodes.push(add(scoresWithRelative, relScores, scoresWithRel));
    scoresWithRelative = scoresWithRel;
  }
  
  // Apply attention mask
  // mask is now [batch, 1, time, time] (2D attention mask created in buildTransformerEncoder)
  // Python: scores.masked_fill(mask == 0, -1e4)
  
  // Create mask condition (where mask == 0, fill with -1e4)
  const zeroConst = addScalar(uniqueName("zero"), 0.0);
  const maskIsZero = uniqueName("mask_is_zero");
  nodes.push(node("Equal", [mask, zeroConst], [maskIsZero]));
  
  const negInf = addScalar(uniqueName("neg_inf"), -10000.0);
  const scoresMasked = uniqueName("scores_masked");
  nodes.push(where(maskIsZero, negInf, scoresWithRelative, scoresMasked));

  // Softmax over last dimension (time_k)
  const attnWeights = uniqueName("attn_weights");
  nodes.push(softmax(scoresMasked, attnWeights, -1));

  // Attention output: attn_weights @ V -> [batch, n_heads, time_q, k_channels]
  let attnOutput = uniqueName("attn_out");
  nodes.push(matmul(attnWeights, vHeads, attnOutput));

  // Add relative values if available
  // Python: if self.window_size: output += self._apply_relative_values(p_attn, t_s)
  if (hasRelativePos && embRelV) {
    const relValues = uniqueName("rel_values");
    buildApplyRelativeValues(
      nodes, initializers, attnWeights, embRelV,
      batchDim, nHeadsConst, windowSize, timeDim,
      relValues, addInt64Const, addFloatConst
    );
    
    const attnWithRel = uniqueName("attn_with_rel");
    nodes.push(add(attnOutput, relValues, attnWithRel));
    attnOutput = attnWithRel;
  }

  // Transpose back: [batch, n_heads, time, k_channels] -> [batch, n_heads, k_channels, time]
  const attnOutT = uniqueName("attn_out_transposed");
  nodes.push(transpose(attnOutput, attnOutT, [0, 1, 3, 2]));
  
  // Reshape to [batch, channels, time]
  const channelsConst = addInt64Const(uniqueName("channels"), [channels], [1]);
  const outShape = uniqueName("out_shape");
  nodes.push(concat([batchDim, channelsConst, timeDim], outShape, 0));
  const attnOutReshaped = uniqueName("attn_out_reshaped");
  nodes.push(reshape(attnOutT, outShape, attnOutReshaped));

  // Output projection
  const output = uniqueName("mha_output");
  nodes.push(conv1d(attnOutReshaped, convOWeight, convOBias, output, 1));

  return output;
}

// =============================================================================
// FFN Builder
// =============================================================================

/**
 * Build Feed-Forward Network with proper same_padding.
 * 
 * Python equivalent (from attentions.py):
 *   x = self.conv_1(self._same_padding(x * x_mask))
 *   x = self._apply_activation(x)  # relu or gelu
 *   x = self.drop(x)
 *   x = self.conv_2(self._same_padding(x * x_mask))
 *   return x * x_mask
 * 
 * _same_padding pads (kernel_size - 1) // 2 on each side
 */
function buildFFN(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  mask: string,
  prefix: string,
  addWeight: (name: string) => string,
  _addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  _addInt64Const: (name: string, values: number[], shape: number[]) => string,
  _addScalar: (name: string, value: number) => string
): string {
  const conv1Weight = addWeight(`${prefix}conv_1.weight`);
  const conv1Bias = addWeight(`${prefix}conv_1.bias`);
  const conv2Weight = addWeight(`${prefix}conv_2.weight`);
  const conv2Bias = addWeight(`${prefix}conv_2.bias`);

  // Get kernel size from weight shape (weight is [out_channels, in_channels/groups, kernel_size])
  // Use config.kernelSize if available, otherwise default to 3
  const kernelSize = config.filterChannels ? config.kernelSize : 3;
  const padding = Math.floor((kernelSize - 1) / 2);

  // x * x_mask before first conv
  const xMasked1 = uniqueName("ffn_x_masked1");
  nodes.push(mul(input, mask, xMasked1));

  // First convolution with same_padding
  const h = uniqueName("ffn_h");
  nodes.push(conv1d(xMasked1, conv1Weight, conv1Bias, h, kernelSize, 1, padding));

  // Activation (ReLU by default, GELU if specified)
  // GELU approximation: x * sigmoid(1.702 * x)
  const hAct = uniqueName("ffn_h_act");
  
  // Check if model uses GELU (some RVC models do)
  // For now, use ReLU as default
  nodes.push(relu(h, hAct));

  // x * x_mask before second conv
  const hMasked = uniqueName("ffn_h_masked");
  nodes.push(mul(hAct, mask, hMasked));

  // Second convolution with same_padding
  const output = uniqueName("ffn_output");
  nodes.push(conv1d(hMasked, conv2Weight, conv2Bias, output, kernelSize, 1, padding));

  // Final mask
  const outputMasked = uniqueName("ffn_output_masked");
  nodes.push(mul(output, mask, outputMasked));

  return outputMasked;
}

// =============================================================================
// Residual Coupling Block Builder (Flow)
// =============================================================================

/**
 * Build ResidualCouplingBlock for normalizing flow.
 * 
 * Python structure (from residuals.py):
 *   self.flows = ModuleList([
 *     ResidualCouplingLayer(...),  # flows.0
 *     Flip(),                       # flows.1
 *     ResidualCouplingLayer(...),  # flows.2
 *     Flip(),                       # flows.3
 *     ...
 *   ])
 * 
 * Forward: for flow in self.flows: x, _ = flow(x, x_mask, g=g, reverse=False)
 * Reverse: for flow in reversed(self.flows): x = flow(x, x_mask, g=g, reverse=True)
 * 
 * Flip operation: torch.flip(x, [1]) - flips along channel dimension
 */
function buildResidualCouplingBlock(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  mask: string,
  g: string,
  reverse: boolean,
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string
): string {
  let x = input;
  // nFlows is typically 4 for RVC models (not stored in config array)
  const nFlows = 4;

  // Create reversed indices for flip operation along channel dimension
  // Dimension size is interChannels (192)
  const reversedIndices = Array.from(
    { length: config.interChannels },
    (_, i) => config.interChannels - 1 - i
  );
  const flipIndicesName = addInt64Const(
    uniqueName("flip_indices"),
    reversedIndices,
    [config.interChannels]
  );

  // Each flow consists of: ResidualCouplingLayer (flows.i*2) + Flip (flows.i*2+1)
  // Total flows = nFlows * 2 (coupling + flip pairs)
  
  if (!reverse) {
    // Forward: process flows in order
    for (let i = 0; i < nFlows; i++) {
      const couplingPrefix = `flow.flows.${i * 2}.`;
      
      // Residual Coupling Layer
      x = buildResidualCouplingLayer(
        nodes,
        initializers,
        weights,
        config,
        x,
        mask,
        g,
        false, // forward
        couplingPrefix,
        addWeight,
        addConstant,
        addInt64Const,
        addScalar
      );

      // Flip: torch.flip(x, [1]) - flip along channel dimension (dim 1)
      const xFlipped = uniqueName("x_flipped");
      nodes.push(flip(x, flipIndicesName, xFlipped, 1));
      x = xFlipped;
    }
  } else {
    // Reverse: process flows in reverse order
    for (let i = nFlows - 1; i >= 0; i--) {
      // Flip first (since we're going in reverse)
      const xFlipped = uniqueName("x_flipped_rev");
      nodes.push(flip(x, flipIndicesName, xFlipped, 1));
      x = xFlipped;
      
      const couplingPrefix = `flow.flows.${i * 2}.`;
      
      // Residual Coupling Layer (reverse)
      x = buildResidualCouplingLayer(
        nodes,
        initializers,
        weights,
        config,
        x,
        mask,
        g,
        true, // reverse
        couplingPrefix,
        addWeight,
        addConstant,
        addInt64Const,
        addScalar
      );
    }
  }

  return x;
}

/**
 * Build ResidualCouplingLayer with full WaveNet.
 * 
 * Python implementation (from residuals.py):
 *   x0, x1 = torch.split(x, [half_channels] * 2, 1)
 *   h = self.pre(x0) * x_mask
 *   h = self.enc(h, x_mask, g=g)  # WaveNet
 *   stats = self.post(h) * x_mask
 *   m = stats  # mean_only=True for RVC
 *   logs = zeros_like(m)
 *   
 *   if not reverse:
 *     x1 = m + x1 * exp(logs) * x_mask  # = m + x1 * 1 * mask = m + x1*mask
 *     x = cat([x0, x1], 1)
 *   else:
 *     x1 = (x1 - m) * exp(-logs) * x_mask  # = (x1 - m) * 1 * mask
 *     x = cat([x0, x1], 1)
 */
function buildResidualCouplingLayer(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  mask: string,
  g: string,
  reverse: boolean,
  prefix: string,
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  _addScalar: (name: string, value: number) => string
): string {
  const channels = config.interChannels;
  const halfChannels = channels / 2;
  const hiddenChannels = config.hiddenChannels;
  
  // Split input in half along channel dimension
  // x0, x1 = torch.split(x, [half_channels] * 2, 1)
  const splitSizes = addInt64Const(uniqueName("split_sizes"), [halfChannels, halfChannels], [2]);
  const x0 = uniqueName("x0");
  const x1 = uniqueName("x1");
  nodes.push(splitWithSizes(input, splitSizes, [x0, x1], 1));

  // h = self.pre(x0) * x_mask
  const preWeight = addWeight(`${prefix}pre.weight`);
  const preBias = addWeight(`${prefix}pre.bias`);
  const hPre = uniqueName("h_pre");
  nodes.push(conv1d(x0, preWeight, preBias, hPre, 1));
  
  const hPreMasked = uniqueName("h_pre_masked");
  nodes.push(mul(hPre, mask, hPreMasked));

  // h = self.enc(h, x_mask, g=g) - Full WaveNet
  // Detect WaveNet layer count from weights (count in_layers)
  let waveNetLayers = 0;
  for (let i = 0; i < 20; i++) {
    const hasLayer = weights.has(`${prefix}enc.in_layers.${i}.bias`) ||
                     weights.has(`${prefix}enc.in_layers.${i}.weight`) ||
                     hasWeightNorm(weights, `${prefix}enc.in_layers.${i}`);
    if (hasLayer) {
      waveNetLayers = i + 1;
    } else {
      break;
    }
  }
  // Fallback to 4 if detection fails
  if (waveNetLayers === 0) waveNetLayers = 4;
  
  // Detect kernel size from the first in_layer weight
  // Weight shape is [out_channels, in_channels, kernel_size]
  let waveNetKernelSize = 5; // default
  const wnKeys = getWeightNormKeys(weights, `${prefix}enc.in_layers.0`);
  if (wnKeys) {
    const weightV = weights.get(wnKeys.weightV);
    if (weightV && weightV.shape.length >= 3) {
      waveNetKernelSize = weightV.shape[2];
    }
  } else if (weights.has(`${prefix}enc.in_layers.0.weight`)) {
    const weight = weights.get(`${prefix}enc.in_layers.0.weight`);
    if (weight && weight.shape.length >= 3) {
      waveNetKernelSize = weight.shape[2];
    }
  }
  
  const h = buildWaveNet(
    nodes,
    initializers,
    weights,
    hPreMasked,
    mask,
    g,
    `${prefix}enc.`,
    hiddenChannels,
    waveNetKernelSize,
    1, // dilationRate - typically 1 for RVC coupling layers
    waveNetLayers,
    addWeight,
    addInt64Const
  );

  // stats = self.post(h) * x_mask
  // For mean_only=True, post outputs half_channels
  const postWeight = addWeight(`${prefix}post.weight`);
  const postBias = addWeight(`${prefix}post.bias`);
  const statsRaw = uniqueName("stats_raw");
  nodes.push(conv1d(h, postWeight, postBias, statsRaw, 1));
  
  const stats = uniqueName("stats");
  nodes.push(mul(statsRaw, mask, stats));
  
  // m = stats (mean_only=True, so no logs split)
  const m = stats;

  // Apply coupling transformation
  let xOut: string;
  if (!reverse) {
    // Forward: x1 = m + x1 * x_mask (since exp(logs) = exp(0) = 1)
    const x1Masked = uniqueName("x1_masked");
    nodes.push(mul(x1, mask, x1Masked));
    
    const x1New = uniqueName("x1_new");
    nodes.push(add(m, x1Masked, x1New));
    
    // Concat back: x = cat([x0, x1], 1)
    xOut = uniqueName("x_coupled");
    nodes.push(concat([x0, x1New], xOut, 1));
  } else {
    // Reverse: x1 = (x1 - m) * x_mask (since exp(-logs) = exp(0) = 1)
    const x1SubM = uniqueName("x1_sub_m");
    nodes.push(sub(x1, m, x1SubM));
    
    const x1New = uniqueName("x1_new");
    nodes.push(mul(x1SubM, mask, x1New));
    
    // Concat back: x = cat([x0, x1], 1)
    xOut = uniqueName("x_coupled");
    nodes.push(concat([x0, x1New], xOut, 1));
  }

  return xOut;
}

// =============================================================================
// HiFiGAN Decoder Builder
// =============================================================================

/**
 * Build HiFiGAN Generator for audio synthesis.
 * 
 * Python structure (from generators/hifigan.py):
 *   x = self.conv_pre(x)
 *   if g is not None:
 *     x = x + self.cond(g)
 *   
 *   for i in range(num_upsamples):
 *     x = leaky_relu(x, 0.1)
 *     x = self.ups[i](x)  # ConvTranspose1d
 *     
 *     # Accumulate resblock outputs and average
 *     xs = None
 *     for j in range(num_kernels):
 *       if xs is None:
 *         xs = self.resblocks[i * num_kernels + j](x)
 *       else:
 *         xs += self.resblocks[i * num_kernels + j](x)
 *     x = xs / num_kernels  # Average
 *   
 *   x = leaky_relu(x)
 *   x = self.conv_post(x)
 *   x = tanh(x)
 *   return x
 */
function buildHiFiGANDecoder(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  f0Input: string | null,
  g: string,
  useF0: boolean,
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string
): string {
  const prefix = "dec.";
  
  // Check if this is an NSF model (has m_source weights for F0-based synthesis)
  const isNSF = useF0 && weights.has(`${prefix}m_source.l_linear.weight`);
  
  // If NSF model with F0, use the GeneratorNSF path
  if (isNSF && f0Input) {
    return buildGeneratorNSF(
      nodes,
      initializers,
      weights,
      config,
      input,
      f0Input,
      g,
      prefix,
      addWeight,
      addConstant,
      addInt64Const,
      addScalar
    );
  }

  // Pre-convolution: conv_pre(x)
  const convPreWeight = addWeight(`${prefix}conv_pre.weight`);
  const convPreBias = addWeight(`${prefix}conv_pre.bias`);
  let x = uniqueName("dec_x");
  nodes.push(conv1d(input, convPreWeight, convPreBias, x, 7, 1, 3));

  // Add global conditioning: x = x + cond(g)
  if (weights.has(`${prefix}cond.weight`)) {
    const condWeight = addWeight(`${prefix}cond.weight`);
    const condBias = weights.has(`${prefix}cond.bias`)
      ? addWeight(`${prefix}cond.bias`)
      : null;
    const gCond = uniqueName("g_cond");
    nodes.push(conv1d(g, condWeight, condBias, gCond, 1));
    const xCond = uniqueName("x_cond");
    nodes.push(add(x, gCond, xCond));
    x = xCond;
  }

  // Upsampling blocks
  const numUpsamples = config.upsampleRates.length;
  const numKernels = config.resblockKernelSizes.length;

  for (let i = 0; i < numUpsamples; i++) {
    // LeakyReLU before upsample
    const xAct = uniqueName("dec_x_act");
    nodes.push(leakyRelu(x, xAct, 0.1));

    // Upsample (ConvTranspose1d) with weight_norm
    // Check for weight_norm decomposition
    const upsPrefix = `${prefix}ups.${i}`;
    const wnKeysUps = getWeightNormKeys(weights, upsPrefix);
    
    let upWeight: string;
    if (wnKeysUps) {
      upWeight = uniqueName(`ups_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysUps.weightG, wnKeysUps.weightV, upWeight);
    } else {
      upWeight = addWeight(`${upsPrefix}.weight`);
    }
    
    const upBias = weights.has(`${prefix}ups.${i}.bias`)
      ? addWeight(`${prefix}ups.${i}.bias`)
      : null;

    const rate = config.upsampleRates[i];
    const kernelSize = config.upsampleKernelSizes[i];
    // Python: if u % 2 == 0: padding = (k - u) // 2, else: padding = u // 2 + u % 2
    const padding = rate % 2 === 0 
      ? Math.floor((kernelSize - rate) / 2)
      : Math.floor(rate / 2) + (rate % 2);
    // Python: output_padding = u % 2 for odd upsampling rates
    const outputPadding = rate % 2;

    const xUp = uniqueName("dec_x_up");
    nodes.push(
      convTranspose1d(xAct, upWeight, upBias, xUp, kernelSize, rate, padding, outputPadding)
    );

    // Residual blocks with PROPER AVERAGING
    // xs = None
    // for j in range(num_kernels):
    //   if xs is None:
    //     xs = resblocks[i * num_kernels + j](x)
    //   else:
    //     xs += resblocks[i * num_kernels + j](x)
    // x = xs / num_kernels
    
    let xs: string | null = null;
    for (let j = 0; j < numKernels; j++) {
      const resIdx = i * numKernels + j;
      const resOut = buildResBlock(
        nodes,
        initializers,
        weights,
        config,
        xUp, // All resblocks take the same upsampled input
        `${prefix}resblocks.${resIdx}.`,
        j,
        addWeight,
        addConstant,
        addInt64Const,
        addScalar
      );
      
      if (xs === null) {
        xs = resOut;
      } else {
        // xs += resblock_output
        const xsNew = uniqueName("xs_acc");
        nodes.push(add(xs, resOut, xsNew));
        xs = xsNew;
      }
    }

    // x = xs / num_kernels (average)
    const numKernelsScalar = addScalar(uniqueName("num_kernels_inv"), 1.0 / numKernels);
    x = uniqueName("dec_x_avg");
    nodes.push(mul(xs!, numKernelsScalar, x));
  }

  // Post-convolution
  const xActFinal = uniqueName("dec_x_act_final");
  nodes.push(leakyRelu(x, xActFinal, 0.1));

  const convPostWeight = addWeight(`${prefix}conv_post.weight`);
  // conv_post typically has no bias (bias=False in Python)
  const audio = uniqueName("audio_raw");
  nodes.push(conv1d(xActFinal, convPostWeight, null, audio, 7, 1, 3));

  // Tanh activation
  const audioFinal = uniqueName("audio_final");
  nodes.push(tanh(audio, audioFinal));

  return audioFinal;
}

// =============================================================================
// ResBlock Builder
// =============================================================================

/**
 * Build ResBlock from residuals.py.
 * 
 * Python structure:
 *   for conv1, conv2 in zip(convs1, convs2):
 *     x_residual = x
 *     x = leaky_relu(x, 0.1)
 *     x = x * x_mask (if mask provided)
 *     x = leaky_relu(conv1(x), 0.1)
 *     x = x * x_mask (if mask provided)
 *     x = conv2(x)
 *     x = x + x_residual
 *   return x * x_mask
 * 
 * Note: HiFiGAN ResBlock doesn't use mask, but encoder/flow ResBlock does
 */
function buildResBlock(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  prefix: string,
  kernelIdx: number,
  addWeight: (name: string) => string,
  _addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  _addInt64Const: (name: string, values: number[], shape: number[]) => string,
  _addScalar: (name: string, value: number) => string
): string {
  let x = input;
  const kernelSize = config.resblockKernelSizes[kernelIdx] || 3;
  const dilations = config.resblockDilationSizes[kernelIdx] || [1, 3, 5];

  for (let i = 0; i < dilations.length; i++) {
    const residual = x;

    // First conv with dilation (weight_norm)
    const xAct1 = uniqueName("resblock_act1");
    nodes.push(leakyRelu(x, xAct1, 0.1));

    // Check for weight_norm decomposition
    const conv1Prefix = `${prefix}convs1.${i}`;
    const wnKeysConv1 = getWeightNormKeys(weights, conv1Prefix);
    
    let conv1Weight: string;
    if (wnKeysConv1) {
      conv1Weight = uniqueName(`resblock_conv1_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysConv1.weightG, wnKeysConv1.weightV, conv1Weight);
    } else {
      conv1Weight = addWeight(`${conv1Prefix}.weight`);
    }
    
    const conv1Bias = weights.has(`${prefix}convs1.${i}.bias`)
      ? addWeight(`${prefix}convs1.${i}.bias`)
      : null;
    const dilation = dilations[i];
    // padding = get_padding(kernel_size, dilation) = (kernel_size * dilation - dilation) // 2
    const padding = Math.floor((kernelSize * dilation - dilation) / 2);
    const h = uniqueName("resblock_h");
    nodes.push(conv1d(xAct1, conv1Weight, conv1Bias, h, kernelSize, 1, padding, dilation));

    // Second conv (dilation=1, weight_norm)
    const hAct = uniqueName("resblock_h_act");
    nodes.push(leakyRelu(h, hAct, 0.1));

    const conv2Prefix = `${prefix}convs2.${i}`;
    const wnKeysConv2 = getWeightNormKeys(weights, conv2Prefix);
    
    let conv2Weight: string;
    if (wnKeysConv2) {
      conv2Weight = uniqueName(`resblock_conv2_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysConv2.weightG, wnKeysConv2.weightV, conv2Weight);
    } else {
      conv2Weight = addWeight(`${conv2Prefix}.weight`);
    }
    
    const conv2Bias = weights.has(`${prefix}convs2.${i}.bias`)
      ? addWeight(`${prefix}convs2.${i}.bias`)
      : null;
    const hConv = uniqueName("resblock_h_conv");
    // Second conv always has dilation=1, padding = (kernel_size - 1) // 2
    const padding2 = Math.floor((kernelSize - 1) / 2);
    nodes.push(conv1d(hAct, conv2Weight, conv2Bias, hConv, kernelSize, 1, padding2, 1));

    // Residual connection: x = x + x_residual
    const xNew = uniqueName("resblock_x_new");
    nodes.push(add(hConv, residual, xNew));
    x = xNew;
  }

  return x;
}

// =============================================================================
// NSF (Neural Source Filter) Components
// =============================================================================

/**
 * Build SineGenerator for F0-based synthesis.
 * 
 * Python implementation (from generators/hifigan.py SineGenerator):
 *   1. Compute voiced/unvoiced mask: uv = (f0 > threshold).float()
 *   2. Compute phase increments: (f0 / sampling_rate) * upsampling_grid
 *   3. Accumulate phase with cumsum and wrap with fmod
 *   4. Generate sine waves: sin(2 * pi * phase)
 *   5. Apply voiced mask and add noise for unvoiced
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param f0 - F0 tensor name [batch, length] or [batch, length, 1]
 * @param upsamplingFactor - Total upsampling factor (e.g., 256)
 * @param samplingRate - Audio sampling rate (e.g., 48000)
 * @param sineAmplitude - Amplitude of sine wave (default 0.1)
 * @param noiseStddev - Stddev of noise (default 0.003)
 * @param voicedThreshold - F0 threshold for voiced (default 0.0)
 * @param addScalar - Helper to add scalar constants
 * @param addInt64Const - Helper to add int64 constants
 * @returns Object with sine_waveforms, voiced_mask, noise tensor names
 */
function buildSineGenerator(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  f0: string,
  upsamplingFactor: number,
  samplingRate: number,
  sineAmplitude: number = 0.1,
  noiseStddev: number = 0.003,
  voicedThreshold: number = 0.0,
  addScalar: (name: string, value: number) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): { sineWaveforms: string; voicedMask: string; noise: string } {
  
  // Step 1: Compute voiced/unvoiced mask
  // uv_mask = (f0 > voiced_threshold).float()
  const voicedThresholdConst = addScalar(uniqueName("voiced_threshold"), voicedThreshold);
  const voicedBool = uniqueName("voiced_bool");
  nodes.push(node("Greater", [f0, voicedThresholdConst], [voicedBool]));
  
  const voicedMaskF32 = uniqueName("voiced_mask_f32");
  nodes.push(cast(voicedBool, voicedMaskF32, OnnxDataType.FLOAT));
  
  // Step 2: Unsqueeze f0: [batch, length] -> [batch, length, 1]
  const f0Expanded = uniqueName("f0_expanded");
  const unsqueezeAxesNeg1 = addInt64Const(uniqueName("unsqueeze_neg1"), [-1], [1]);
  nodes.push(unsqueeze(f0, unsqueezeAxesNeg1, f0Expanded));
  
  // Step 3: Create upsampling grid [1, 2, ..., upsamplingFactor]
  const oneConstInt = addInt64Const(uniqueName("one_int"), [1], []);
  const upFactorPlusOne = addInt64Const(uniqueName("up_factor_plus_one"), [upsamplingFactor + 1], []);
  const upsamplingGrid = uniqueName("upsampling_grid");
  nodes.push(range(oneConstInt, upFactorPlusOne, oneConstInt, upsamplingGrid));
  
  // Cast to float for computation
  const upsamplingGridF32 = uniqueName("upsampling_grid_f32");
  nodes.push(cast(upsamplingGrid, upsamplingGridF32, OnnxDataType.FLOAT));
  
  // Step 4: Calculate phase increments
  // phase_increments = (f0 / sampling_rate) * upsampling_grid
  // f0Expanded: [batch, length, 1], upsamplingGridF32: [upsamplingFactor]
  // Result: [batch, length, upsamplingFactor]
  const srConst = addScalar(uniqueName("sampling_rate"), samplingRate);
  const f0Normalized = uniqueName("f0_normalized");
  nodes.push(div(f0Expanded, srConst, f0Normalized));
  
  const phaseIncrementsRaw = uniqueName("phase_increments_raw");
  nodes.push(mul(f0Normalized, upsamplingGridF32, phaseIncrementsRaw));
  
  // Step 5: Compute cumulative phase across frames (FULL IMPLEMENTATION)
  // Python:
  //   phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
  //   cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0)
  //   phase_increments += F.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")
  
  // Step 5a: Extract phase_increments[:, :-1, -1:]
  // This gets the last element of the upsampling grid for all frames except the last
  // Shape: [batch, length-1, 1]
  const sliceStart5a = addInt64Const(uniqueName("slice_start_5a"), [0, 0, upsamplingFactor - 1], [3]);
  const sliceEnd5a = addInt64Const(uniqueName("slice_end_5a"), [2147483647, -1, 2147483647], [3]);
  const sliceAxes5a = addInt64Const(uniqueName("slice_axes_5a"), [0, 1, 2], [3]);
  const sliceSteps5a = addInt64Const(uniqueName("slice_steps_5a"), [1, 1, 1], [3]);
  const phaseLastCol = uniqueName("phase_last_col");
  nodes.push(slice(phaseIncrementsRaw, sliceStart5a, sliceEnd5a, sliceAxes5a, sliceSteps5a, phaseLastCol));
  
  // Step 5b: phase_remainder = fmod(phase_last_col + 0.5, 1.0) - 0.5
  // This wraps the phase to [-0.5, 0.5) range
  const halfConst = addScalar(uniqueName("half"), 0.5);
  const oneConstF32 = addScalar(uniqueName("one_f32"), 1.0);
  
  const phasePlusHalf = uniqueName("phase_plus_half");
  nodes.push(add(phaseLastCol, halfConst, phasePlusHalf));
  
  const phaseModOne = uniqueName("phase_mod_one");
  nodes.push(node("Mod", [phasePlusHalf, oneConstF32], [phaseModOne], [
    { name: "fmod", type: "INT", intValue: 1n }
  ]));
  
  const phaseRemainder = uniqueName("phase_remainder");
  nodes.push(sub(phaseModOne, halfConst, phaseRemainder));
  
  // Step 5c: cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0)
  const cumulativePhaseRaw = uniqueName("cumulative_phase_raw");
  nodes.push(node("CumSum", [phaseRemainder, addInt64Const(uniqueName("cumsum_axis"), [1], [])], [cumulativePhaseRaw]));
  
  const cumulativePhase = uniqueName("cumulative_phase");
  nodes.push(node("Mod", [cumulativePhaseRaw, oneConstF32], [cumulativePhase], [
    { name: "fmod", type: "INT", intValue: 1n }
  ]));
  
  // Step 5d: Pad cumulative_phase with 0 at the start: (0, 0, 1, 0) means pad 1 on left of dim 1
  // cumulative_phase: [batch, length-1, 1] -> [batch, length, 1]
  // ONNX Pad uses format [dim0_start, dim1_start, dim2_start, dim0_end, dim1_end, dim2_end]
  const padConst = addInt64Const(uniqueName("pad_const"), [0, 1, 0, 0, 0, 0], [6]);
  const zeroConstF32 = addScalar(uniqueName("zero_f32"), 0.0);
  const cumulativePhasePadded = uniqueName("cumulative_phase_padded");
  nodes.push(node("Pad", [cumulativePhase, padConst, zeroConstF32], [cumulativePhasePadded]));
  
  // Step 5e: phase_increments += cumulative_phase_padded (broadcasts [batch, length, 1] to [batch, length, upsamplingFactor])
  const phaseIncrements = uniqueName("phase_increments");
  nodes.push(add(phaseIncrementsRaw, cumulativePhasePadded, phaseIncrements));
  
  // Step 6: Reshape to [batch, length * upsamplingFactor, 1]
  // Python: phase_increments = phase_increments.reshape(batch_size, -1, 1)
  const batchIdxSine = addInt64Const(uniqueName("batch_idx_sine"), [0], [1]);
  const phaseShape = uniqueName("phase_shape");
  nodes.push(shape(phaseIncrements, phaseShape));
  const batchDimPhase = uniqueName("batch_dim_phase");
  nodes.push(gather(phaseShape, batchIdxSine, batchDimPhase, 0));
  
  const negOneConst = addInt64Const(uniqueName("neg_one"), [-1], [1]);
  const oneConstDim = addInt64Const(uniqueName("one_dim"), [1], [1]);
  const reshapePhaseShape = uniqueName("reshape_phase_shape");
  nodes.push(concat([batchDimPhase, negOneConst, oneConstDim], reshapePhaseShape, 0));
  
  const phaseIncrementsFlat = uniqueName("phase_increments_flat");
  nodes.push(reshape(phaseIncrements, reshapePhaseShape, phaseIncrementsFlat));
  
  // Step 7: Harmonic scaling (for harmonic_num > 0)
  // Python: harmonic_scale = arange(1, waveform_dim + 1).reshape(1, 1, -1)
  //         phase_increments *= harmonic_scale
  // For RVC, harmonic_num=0, so waveform_dim=1, harmonic_scale=[1], no-op
  // We include it for completeness but it's effectively multiplying by 1
  
  // Step 8: Random phase offset (for harmonics > 0)
  // Python: random_phase = torch.rand(1, 1, waveform_dim)
  //         random_phase[..., 0] = 0  # Fundamental has no offset
  //         phase_increments += random_phase
  // For RVC with harmonic_num=0, this is just adding 0 to fundamental, so skip
  
  // Step 9: Generate sine waves
  // sine_waves = sin(2 * pi * phase_increments) * sine_amplitude
  const twoPi = addScalar(uniqueName("two_pi"), 2 * Math.PI);
  const phaseRadians = uniqueName("phase_radians");
  nodes.push(mul(phaseIncrementsFlat, twoPi, phaseRadians));
  
  const sineRaw = uniqueName("sine_raw");
  nodes.push(sin(phaseRadians, sineRaw));
  
  const sineAmpConst = addScalar(uniqueName("sine_amplitude"), sineAmplitude);
  const sineScaled = uniqueName("sine_scaled");
  nodes.push(mul(sineRaw, sineAmpConst, sineScaled));
  
  // Step 10: Upsample voiced mask to match sine wave length
  // Python: voiced_mask = interpolate(voiced_mask.transpose(2, 1), scale_factor, mode='nearest').transpose(2, 1)
  // voicedMaskF32: [batch, length]
  // 1. Unsqueeze to [batch, length, 1]
  // 2. Transpose to [batch, 1, length]
  // 3. Resize to [batch, 1, length * upsamplingFactor]
  // 4. Transpose back to [batch, length * upsamplingFactor, 1]
  
  const voicedMaskExpanded = uniqueName("voiced_mask_expanded");
  nodes.push(unsqueeze(voicedMaskF32, unsqueezeAxesNeg1, voicedMaskExpanded));
  
  // Transpose [batch, length, 1] -> [batch, 1, length]
  const voicedMaskT = uniqueName("voiced_mask_transposed");
  nodes.push(transpose(voicedMaskExpanded, voicedMaskT, [0, 2, 1]));
  
  // Create scale factors for Resize: [1, 1, upsamplingFactor]
  const scalesData = new Float32Array([1.0, 1.0, upsamplingFactor]);
  const scalesConst = uniqueName("resize_scales");
  initializers.push(initializer(scalesConst, {
    data: scalesData,
    shape: [3],
    dtype: "float32",
  }));
  
  // Resize [batch, 1, length] -> [batch, 1, length * upsamplingFactor]
  const voicedMaskResized = uniqueName("voiced_mask_resized");
  nodes.push(node("Resize", [voicedMaskT, "", scalesConst], [voicedMaskResized], [
    { name: "mode", type: "STRING", stringValue: "nearest" },
    { name: "coordinate_transformation_mode", type: "STRING", stringValue: "asymmetric" },
    { name: "nearest_mode", type: "STRING", stringValue: "floor" },
  ]));
  
  // Transpose back [batch, 1, length * upsamplingFactor] -> [batch, length * upsamplingFactor, 1]
  const voicedMaskUpsampled = uniqueName("voiced_mask_upsampled");
  nodes.push(transpose(voicedMaskResized, voicedMaskUpsampled, [0, 2, 1]));
  
  // Step 11: Compute noise amplitude
  // noise_amplitude = voiced_mask * noise_stddev + (1 - voiced_mask) * (sine_amplitude / 3)
  const noiseStdConst = addScalar(uniqueName("noise_stddev"), noiseStddev);
  const unvoicedAmp = addScalar(uniqueName("unvoiced_amp"), sineAmplitude / 3);
  
  const voicedNoiseAmp = uniqueName("voiced_noise_amp");
  nodes.push(mul(voicedMaskUpsampled, noiseStdConst, voicedNoiseAmp));
  
  const oneMinusVoiced = uniqueName("one_minus_voiced");
  nodes.push(sub(oneConstF32, voicedMaskUpsampled, oneMinusVoiced));
  
  const unvoicedNoiseAmp = uniqueName("unvoiced_noise_amp");
  nodes.push(mul(oneMinusVoiced, unvoicedAmp, unvoicedNoiseAmp));
  
  const noiseAmplitude = uniqueName("noise_amplitude");
  nodes.push(add(voicedNoiseAmp, unvoicedNoiseAmp, noiseAmplitude));
  
  // Step 12: Generate noise
  const noiseRandom = uniqueName("noise_random");
  nodes.push(randomNormalLike(sineScaled, noiseRandom, 0.0, 1.0));
  
  const noise = uniqueName("noise");
  nodes.push(mul(noiseAmplitude, noiseRandom, noise));
  
  // Step 13: Combine sine waves and noise
  // sine_waveforms = sine_waves * voiced_mask + noise
  const sineVoiced = uniqueName("sine_voiced");
  nodes.push(mul(sineScaled, voicedMaskUpsampled, sineVoiced));
  
  const sineWaveforms = uniqueName("sine_waveforms");
  nodes.push(add(sineVoiced, noise, sineWaveforms));
  
  return {
    sineWaveforms,
    voicedMask: voicedMaskUpsampled,
    noise,
  };
}

/**
 * Build SourceModuleHnNSF for harmonic-plus-noise synthesis.
 * 
 * Python implementation:
 *   - Uses SineGenerator to create harmonic source signal
 *   - Applies learnable linear layers to shape the source
 *   - Outputs source signal for HiFiGAN integration
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param weights - Weight map
 * @param f0 - F0 tensor name
 * @param upsamplingFactor - Total upsampling factor
 * @param samplingRate - Audio sampling rate
 * @param prefix - Weight prefix (e.g., "dec.m_source.")
 * @param addWeight - Helper to add weights
 * @param addScalar - Helper to add scalars
 * @param addInt64Const - Helper to add int64 constants
 * @returns Source signal tensor name
 */
function buildSourceModuleHnNSF(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  f0: string,
  upsamplingFactor: number,
  samplingRate: number,
  prefix: string,
  addWeight: (name: string) => string,
  addScalar: (name: string, value: number) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string
): string {
  // Generate sine source
  const { sineWaveforms } = buildSineGenerator(
    nodes,
    initializers,
    f0,
    upsamplingFactor,
    samplingRate,
    0.1,  // sine_amplitude
    0.003, // noise_stddev
    0.0,   // voiced_threshold
    addScalar,
    addInt64Const
  );
  
  // Apply learnable transformation if weights exist
  // l_linear = nn.Linear(sine_wave_dim, out_dim)
  // Python: sine_merge = self.l_tanh(self.l_linear(sine_wavs))
  if (weights.has(`${prefix}l_linear.weight`)) {
    const linearWeight = addWeight(`${prefix}l_linear.weight`);
    const linearBias = weights.has(`${prefix}l_linear.bias`) 
      ? addWeight(`${prefix}l_linear.bias`)
      : null;
    
    const linearOut = uniqueName("nsf_linear_out");
    nodes.push(...linearNodes(sineWaveforms, linearWeight, linearBias, linearOut));
    
    // Apply tanh activation (l_tanh in Python)
    const tanhOut = uniqueName("nsf_tanh_out");
    nodes.push(tanh(linearOut, tanhOut));
    
    // Transpose from [batch, time, channels] to [batch, channels, time]
    // This is needed because Conv1d expects [B, C, T] format
    const source = uniqueName("nsf_source");
    nodes.push(transpose(tanhOut, source, [0, 2, 1]));
    return source;
  }
  
  // If no transformation, return raw sine waveforms
  // Also need to transpose here since sineWaveforms is [B, T, 1]
  const sourceTransposed = uniqueName("nsf_source_transposed");
  nodes.push(transpose(sineWaveforms, sourceTransposed, [0, 2, 1]));
  return sourceTransposed;
}

/**
 * Build GeneratorNSF - HiFiGAN with NSF source module.
 * 
 * This is a variant of HiFiGAN that incorporates F0-based source signals
 * for better pitch-controllable synthesis.
 * 
 * Python implementation:
 *   - F0 is upsampled to match audio resolution
 *   - Source module generates harmonic source signal
 *   - HiFiGAN processes both latent z and source signal
 * 
 * @param nodes - Array to add nodes to
 * @param initializers - Array to add initializers to
 * @param weights - Weight map
 * @param config - Model config
 * @param input - Input latent tensor [batch, channels, time]
 * @param f0 - F0 tensor [batch, f0_len]
 * @param g - Global conditioning tensor
 * @param prefix - Weight prefix
 * @param addWeight - Helper to add weights
 * @param addConstant - Helper to add constants
 * @param addInt64Const - Helper to add int64 constants
 * @param addScalar - Helper to add scalars
 * @returns Audio output tensor name
 */
function buildGeneratorNSF(
  nodes: OnnxNode[],
  initializers: OnnxInitializer[],
  weights: Map<string, TensorData>,
  config: ParsedCheckpoint["config"],
  input: string,
  f0: string,
  g: string,
  prefix: string,
  addWeight: (name: string) => string,
  addConstant: (
    name: string,
    data: number[] | Float32Array,
    shape: number[]
  ) => string,
  addInt64Const: (name: string, values: number[], shape: number[]) => string,
  addScalar: (name: string, value: number) => string
): string {
  // Calculate total upsampling factor
  const totalUpsample = config.upsampleRates.reduce((a, b) => a * b, 1);
  
  // Generate NSF source if model has source module
  let source: string | null = null;
  if (weights.has(`${prefix}m_source.l_linear.weight`)) {
    source = buildSourceModuleHnNSF(
      nodes,
      initializers,
      weights,
      f0,
      totalUpsample,
      config.sr,
      `${prefix}m_source.`,
      addWeight,
      addScalar,
      addInt64Const
    );
  }
  
  // Pre-convolution
  const convPreWeight = addWeight(`${prefix}conv_pre.weight`);
  const convPreBias = addWeight(`${prefix}conv_pre.bias`);
  let x = uniqueName("nsf_x");
  nodes.push(conv1d(input, convPreWeight, convPreBias, x, 7, 1, 3));
  
  // Add conditioning
  if (weights.has(`${prefix}cond.weight`)) {
    const condWeight = addWeight(`${prefix}cond.weight`);
    const condBias = weights.has(`${prefix}cond.bias`)
      ? addWeight(`${prefix}cond.bias`)
      : null;
    const gCond = uniqueName("nsf_g_cond");
    nodes.push(conv1d(g, condWeight, condBias, gCond, 1));
    const xCond = uniqueName("nsf_x_cond");
    nodes.push(add(x, gCond, xCond));
    x = xCond;
  }
  
  // Upsampling blocks with optional source injection
  const numUpsamples = config.upsampleRates.length;
  const numKernels = config.resblockKernelSizes.length;
  
  for (let i = 0; i < numUpsamples; i++) {
    // LeakyReLU
    const xAct = uniqueName("nsf_x_act");
    nodes.push(leakyRelu(x, xAct, 0.1));
    
    // Upsample
    const nsfUpsPrefix = `${prefix}ups.${i}`;
    const wnKeysNsfUps = getWeightNormKeys(weights, nsfUpsPrefix);
    let upWeight: string;
    if (wnKeysNsfUps) {
      upWeight = uniqueName(`nsf_ups_${i}_weight`);
      buildWeightNormReconstruction(nodes, initializers, weights, wnKeysNsfUps.weightG, wnKeysNsfUps.weightV, upWeight);
    } else {
      upWeight = addWeight(`${nsfUpsPrefix}.weight`);
    }
    
    const upBias = weights.has(`${prefix}ups.${i}.bias`)
      ? addWeight(`${prefix}ups.${i}.bias`)
      : null;
    
    const rate = config.upsampleRates[i];
    const kernelSize = config.upsampleKernelSizes[i];
    // Python: if u % 2 == 0: padding = (k - u) // 2, else: padding = u // 2 + u % 2
    const padding = rate % 2 === 0 
      ? Math.floor((kernelSize - rate) / 2)
      : Math.floor(rate / 2) + (rate % 2);
    // Python: output_padding = u % 2 for odd upsampling rates
    const outputPadding = rate % 2;
    
    const xUp = uniqueName("nsf_x_up");
    nodes.push(convTranspose1d(xAct, upWeight, upBias, xUp, kernelSize, rate, padding, outputPadding));
    
    // Add source signal if available (noise shaping)
    // In NSF, source is added at specific upsampling stages
    if (source && weights.has(`${prefix}noise_convs.${i}.weight`)) {
      const noiseConvWeight = addWeight(`${prefix}noise_convs.${i}.weight`);
      const noiseConvBias = weights.has(`${prefix}noise_convs.${i}.bias`)
        ? addWeight(`${prefix}noise_convs.${i}.bias`)
        : null;
      
      // Get kernel size from weight shape: [out_channels, in_channels, kernel_size]
      const noiseWeightTensor = weights.get(`${prefix}noise_convs.${i}.weight`)!;
      const noiseKernelSize = noiseWeightTensor.shape[2];
      
      // Calculate stride: product of remaining upsample rates after this stage
      // Python: stride_f0s[i] = math.prod(upsample_rates[i + 1 :])
      const remainingRates = config.upsampleRates.slice(i + 1);
      const noiseStride = remainingRates.reduce((a: number, b: number) => a * b, 1);
      
      // Padding matches Python: padding = 0 if stride == 1 else (kernel - stride) // 2
      const noisePadding = noiseStride === 1 ? 0 : Math.floor((noiseKernelSize - noiseStride) / 2);
      
      const sourceConv = uniqueName("nsf_source_conv");
      nodes.push(conv1d(source, noiseConvWeight, noiseConvBias, sourceConv, noiseKernelSize, noiseStride, noisePadding));
      
      const xWithSource = uniqueName("nsf_x_with_source");
      nodes.push(add(xUp, sourceConv, xWithSource));
      x = xWithSource;
    } else {
      x = xUp;
    }
    
    // ResBlocks with averaging
    let xs: string | null = null;
    for (let j = 0; j < numKernels; j++) {
      const resIdx = i * numKernels + j;
      const resOut = buildResBlock(
        nodes,
        initializers,
        weights,
        config,
        x,
        `${prefix}resblocks.${resIdx}.`,
        j,
        addWeight,
        addConstant,
        addInt64Const,
        addScalar
      );
      
      if (xs === null) {
        xs = resOut;
      } else {
        const xsNew = uniqueName("nsf_xs_acc");
        nodes.push(add(xs, resOut, xsNew));
        xs = xsNew;
      }
    }
    
    const numKernelsInv = addScalar(uniqueName("nsf_num_kernels_inv"), 1.0 / numKernels);
    x = uniqueName("nsf_x_avg");
    nodes.push(mul(xs!, numKernelsInv, x));
  }
  
  // Post-convolution
  const xActFinal = uniqueName("nsf_x_act_final");
  nodes.push(leakyRelu(x, xActFinal, 0.1));
  
  const convPostWeight = addWeight(`${prefix}conv_post.weight`);
  const audio = uniqueName("nsf_audio_raw");
  nodes.push(conv1d(xActFinal, convPostWeight, null, audio, 7, 1, 3));
  
  const audioFinal = uniqueName("nsf_audio_final");
  nodes.push(tanh(audio, audioFinal));
  
  return audioFinal;
}
