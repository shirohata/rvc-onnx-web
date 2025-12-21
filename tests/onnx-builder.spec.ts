/**
 * ONNX Builder Unit Tests
 * 
 * Tests for ONNX graph building utilities and helper functions.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  uniqueName,
  resetNameCounter,
  valueInfo,
  initializer,
  node,
  attrInt,
  attrInts,
  attrFloat,
  attrFloats,
  attrString,
  conv1d,
  convTranspose1d,
  linearNodes,
  matmul,
  layerNorm,
  relu,
  leakyRelu,
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
  less,
  cast,
  range,
  where,
  reduceSum,
  cumsum,
  mod,
  floor,
  ceil,
  abs
} from "../src/onnx-builder.js";
import { OnnxDataType } from "../src/types.js";

describe("ONNX Builder", () => {
  beforeEach(() => {
    resetNameCounter();
  });

  describe("Name Generation", () => {
    it("should generate unique names", () => {
      const name1 = uniqueName("test");
      const name2 = uniqueName("test");
      const name3 = uniqueName("other");
      
      expect(name1).toBe("test_0");
      expect(name2).toBe("test_1");
      expect(name3).toBe("other_2");
    });

    it("should reset counter", () => {
      uniqueName("test");
      uniqueName("test");
      resetNameCounter();
      const name = uniqueName("test");
      expect(name).toBe("test_0");
    });
  });

  describe("ValueInfo Creation", () => {
    it("should create value info with fixed dimensions", () => {
      const info = valueInfo("input", OnnxDataType.FLOAT, [1, 3, 224, 224]);
      
      expect(info.name).toBe("input");
      expect(info.elemType).toBe(OnnxDataType.FLOAT);
      expect(info.shape).toHaveLength(4);
      expect(info.shape[0]).toEqual({ dimValue: 1n });
      expect(info.shape[3]).toEqual({ dimValue: 224n });
    });

    it("should create value info with dynamic dimensions", () => {
      const info = valueInfo("input", OnnxDataType.FLOAT, ["batch", 3, "height", "width"]);
      
      expect(info.shape[0]).toEqual({ dimParam: "batch" });
      expect(info.shape[1]).toEqual({ dimValue: 3n });
      expect(info.shape[2]).toEqual({ dimParam: "height" });
      expect(info.shape[3]).toEqual({ dimParam: "width" });
    });
  });

  describe("Initializer Creation", () => {
    it("should create initializer from tensor data", () => {
      const data = {
        data: new Float32Array([1, 2, 3, 4]),
        shape: [2, 2],
        dtype: "float32" as const
      };
      
      const init = initializer("weights", data);
      
      expect(init.name).toBe("weights");
      expect(init.data).toBe(data);
    });
  });

  describe("Node Creation", () => {
    it("should create basic node", () => {
      const n = node("Add", ["a", "b"], ["c"]);
      
      expect(n.opType).toBe("Add");
      expect(n.inputs).toEqual(["a", "b"]);
      expect(n.outputs).toEqual(["c"]);
      expect(n.attributes).toEqual([]);
      expect(n.name).toMatch(/^add_/);
    });

    it("should create node with custom name", () => {
      const n = node("Add", ["a", "b"], ["c"], [], "my_add");
      
      expect(n.name).toBe("my_add");
    });

    it("should create node with attributes", () => {
      const attrs = [
        { name: "alpha", type: "FLOAT" as const, floatValue: 0.5 }
      ];
      const n = node("LeakyRelu", ["x"], ["y"], attrs);
      
      expect(n.attributes).toHaveLength(1);
      expect(n.attributes[0].name).toBe("alpha");
    });
  });

  describe("Conv Operations", () => {
    it("should create Conv1d without bias", () => {
      const n = conv1d("input", "weight", null, "output", 3, 1, 1, 1, 1);
      
      expect(n.opType).toBe("Conv");
      expect(n.inputs).toEqual(["input", "weight"]);
      expect(n.outputs).toEqual(["output"]);
      
      const kernelAttr = n.attributes.find(a => a.name === "kernel_shape");
      expect(kernelAttr?.intsValue).toEqual([3n]);
    });

    it("should create Conv1d with bias", () => {
      const n = conv1d("input", "weight", "bias", "output", 3);
      
      expect(n.inputs).toEqual(["input", "weight", "bias"]);
    });

    it("should create ConvTranspose1d", () => {
      const n = convTranspose1d("input", "weight", "bias", "output", 4, 2, 1, 0);
      
      expect(n.opType).toBe("ConvTranspose");
      expect(n.inputs).toEqual(["input", "weight", "bias"]);
      
      const strideAttr = n.attributes.find(a => a.name === "strides");
      expect(strideAttr?.intsValue).toEqual([2n]);
    });
  });

  describe("Linear Operations", () => {
    it("should create linear layer with bias", () => {
      const nodes = linearNodes("input", "weight", "bias", "output");
      
      expect(nodes).toHaveLength(3); // Transpose, MatMul, Add
      expect(nodes[0].opType).toBe("Transpose");
      expect(nodes[1].opType).toBe("MatMul");
      expect(nodes[2].opType).toBe("Add");
    });

    it("should create linear layer without bias", () => {
      const nodes = linearNodes("input", "weight", null, "output");
      
      expect(nodes).toHaveLength(2); // Transpose, MatMul
      expect(nodes[1].outputs[0]).toBe("output");
    });

    it("should create MatMul", () => {
      const n = matmul("a", "b", "c");
      
      expect(n.opType).toBe("MatMul");
      expect(n.inputs).toEqual(["a", "b"]);
      expect(n.outputs).toEqual(["c"]);
    });
  });

  describe("Normalization Operations", () => {
    it("should create LayerNorm", () => {
      const n = layerNorm("input", "scale", "bias", "output", -1, 1e-5);
      
      expect(n.opType).toBe("LayerNormalization");
      expect(n.inputs).toEqual(["input", "scale", "bias"]);
      expect(n.outputs).toEqual(["output"]);
      
      const axisAttr = n.attributes.find(a => a.name === "axis");
      expect(axisAttr?.intValue).toBe(-1n);
      
      const epsAttr = n.attributes.find(a => a.name === "epsilon");
      expect(epsAttr?.floatValue).toBe(1e-5);
    });
  });

  describe("Activation Operations", () => {
    it("should create Relu", () => {
      const n = relu("input", "output");
      expect(n.opType).toBe("Relu");
      expect(n.inputs).toEqual(["input"]);
      expect(n.outputs).toEqual(["output"]);
    });

    it("should create LeakyRelu", () => {
      const n = leakyRelu("input", "output", 0.2);
      expect(n.opType).toBe("LeakyRelu");
      
      const alphaAttr = n.attributes.find(a => a.name === "alpha");
      expect(alphaAttr?.floatValue).toBe(0.2);
    });

    it("should create Sigmoid", () => {
      const n = sigmoid("input", "output");
      expect(n.opType).toBe("Sigmoid");
    });

    it("should create Tanh", () => {
      const n = tanh("input", "output");
      expect(n.opType).toBe("Tanh");
    });

    it("should create Softmax", () => {
      const n = softmax("input", "output", -1);
      expect(n.opType).toBe("Softmax");
      
      const axisAttr = n.attributes.find(a => a.name === "axis");
      expect(axisAttr?.intValue).toBe(-1n);
    });
  });

  describe("Arithmetic Operations", () => {
    it("should create Add", () => {
      const n = add("a", "b", "c");
      expect(n.opType).toBe("Add");
      expect(n.inputs).toEqual(["a", "b"]);
      expect(n.outputs).toEqual(["c"]);
    });

    it("should create Mul", () => {
      const n = mul("a", "b", "c");
      expect(n.opType).toBe("Mul");
    });

    it("should create Sub", () => {
      const n = sub("a", "b", "c");
      expect(n.opType).toBe("Sub");
    });

    it("should create Div", () => {
      const n = div("a", "b", "c");
      expect(n.opType).toBe("Div");
    });

    it("should create Exp", () => {
      const n = exp("input", "output");
      expect(n.opType).toBe("Exp");
    });
  });

  describe("Tensor Shape Operations", () => {
    it("should create Transpose", () => {
      const n = transpose("input", "output", [0, 2, 1]);
      expect(n.opType).toBe("Transpose");
      
      const permAttr = n.attributes.find(a => a.name === "perm");
      expect(permAttr?.intsValue).toEqual([0n, 2n, 1n]);
    });

    it("should create Reshape", () => {
      const n = reshape("input", "shape", "output");
      expect(n.opType).toBe("Reshape");
      expect(n.inputs).toEqual(["input", "shape"]);
    });

    it("should create Unsqueeze", () => {
      const n = unsqueeze("input", "axes", "output");
      expect(n.opType).toBe("Unsqueeze");
      expect(n.inputs).toEqual(["input", "axes"]);
    });

    it("should create Concat", () => {
      const n = concat(["a", "b", "c"], "output", 1);
      expect(n.opType).toBe("Concat");
      expect(n.inputs).toEqual(["a", "b", "c"]);
      
      const axisAttr = n.attributes.find(a => a.name === "axis");
      expect(axisAttr?.intValue).toBe(1n);
    });

    it("should create Split with sizes", () => {
      const n = splitWithSizes("input", "sizes", ["out1", "out2"], 0);
      expect(n.opType).toBe("Split");
      expect(n.inputs).toEqual(["input", "sizes"]);
      expect(n.outputs).toEqual(["out1", "out2"]);
    });
  });

  describe("Indexing Operations", () => {
    it("should create Gather", () => {
      const n = gather("data", "indices", "output", 0);
      expect(n.opType).toBe("Gather");
      expect(n.inputs).toEqual(["data", "indices"]);
      
      const axisAttr = n.attributes.find(a => a.name === "axis");
      expect(axisAttr?.intValue).toBe(0n);
    });

    it("should create Slice", () => {
      const n = slice("input", "starts", "ends", "axes", "steps", "output");
      expect(n.opType).toBe("Slice");
      expect(n.inputs).toEqual(["input", "starts", "ends", "axes", "steps"]);
    });
  });

  describe("Padding Operations", () => {
    it("should create Pad without constant value", () => {
      const n = pad("input", "pads", "output", "constant");
      expect(n.opType).toBe("Pad");
      expect(n.inputs).toEqual(["input", "pads"]);
      
      const modeAttr = n.attributes.find(a => a.name === "mode");
      expect(modeAttr?.stringValue).toBe("constant");
    });

    it("should create Pad with constant value", () => {
      const n = pad("input", "pads", "output", "constant", "value");
      expect(n.inputs).toEqual(["input", "pads", "value"]);
    });

    it("should create Pad with reflect mode", () => {
      const n = pad("input", "pads", "output", "reflect");
      
      const modeAttr = n.attributes.find(a => a.name === "mode");
      expect(modeAttr?.stringValue).toBe("reflect");
    });
  });

  describe("Math Operations", () => {
    it("should create Sin", () => {
      const n = sin("input", "output");
      expect(n.opType).toBe("Sin");
    });

    it("should create Floor", () => {
      const n = floor("input", "output");
      expect(n.opType).toBe("Floor");
    });

    it("should create Ceil", () => {
      const n = ceil("input", "output");
      expect(n.opType).toBe("Ceil");
    });

    it("should create Abs", () => {
      const n = abs("input", "output");
      expect(n.opType).toBe("Abs");
    });

    it("should create Mod", () => {
      const n = mod("a", "b", "output", true);
      expect(n.opType).toBe("Mod");
      expect(n.inputs).toEqual(["a", "b"]);
      
      const fmodAttr = n.attributes.find(a => a.name === "fmod");
      expect(fmodAttr?.intValue).toBe(1n);
    });
  });

  describe("Comparison and Logic Operations", () => {
    it("should create Less", () => {
      const n = less("a", "b", "output");
      expect(n.opType).toBe("Less");
      expect(n.inputs).toEqual(["a", "b"]);
    });

    it("should create Cast", () => {
      const n = cast("input", "output", OnnxDataType.INT64);
      expect(n.opType).toBe("Cast");
      
      const toAttr = n.attributes.find(a => a.name === "to");
      expect(toAttr?.intValue).toBe(BigInt(OnnxDataType.INT64));
    });

    it("should create Where", () => {
      const n = where("condition", "x", "y", "output");
      expect(n.opType).toBe("Where");
      expect(n.inputs).toEqual(["condition", "x", "y"]);
    });

    it("should create Range", () => {
      const n = range("start", "limit", "delta", "output");
      expect(n.opType).toBe("Range");
      expect(n.inputs).toEqual(["start", "limit", "delta"]);
    });
  });

  describe("Reduction Operations", () => {
    it("should create ReduceSum with keepdims", () => {
      const n = reduceSum("input", "axes", "output", true);
      expect(n.opType).toBe("ReduceSum");
      expect(n.inputs).toEqual(["input", "axes"]);
      
      const keepdimsAttr = n.attributes.find(a => a.name === "keepdims");
      expect(keepdimsAttr?.intValue).toBe(1n);
    });

    it("should create ReduceSum without keepdims", () => {
      const n = reduceSum("input", "axes", "output", false);
      
      const keepdimsAttr = n.attributes.find(a => a.name === "keepdims");
      expect(keepdimsAttr?.intValue).toBe(0n);
    });

    it("should create Cumsum", () => {
      const n = cumsum("input", "axis", "output", false, false);
      expect(n.opType).toBe("CumSum");
      expect(n.inputs).toEqual(["input", "axis"]);
      
      const exclusiveAttr = n.attributes.find(a => a.name === "exclusive");
      expect(exclusiveAttr?.intValue).toBe(0n);
      
      const reverseAttr = n.attributes.find(a => a.name === "reverse");
      expect(reverseAttr?.intValue).toBe(0n);
    });

    it("should create Cumsum with exclusive and reverse", () => {
      const n = cumsum("input", "axis", "output", true, true);
      
      const exclusiveAttr = n.attributes.find(a => a.name === "exclusive");
      expect(exclusiveAttr?.intValue).toBe(1n);
      
      const reverseAttr = n.attributes.find(a => a.name === "reverse");
      expect(reverseAttr?.intValue).toBe(1n);
    });
  });

  describe("Attribute Helpers", () => {
    it("should create int attribute", () => {
      const attr = attrInt("test", 42);
      expect(attr.name).toBe("test");
      expect(attr.intValue).toBe(42n);
      expect(attr.type).toBe("INT");
    });

    it("should create int attribute from bigint", () => {
      const attr = attrInt("test", 42n);
      expect(attr.name).toBe("test");
      expect(attr.intValue).toBe(42n);
      expect(attr.type).toBe("INT");
    });

    it("should create ints attribute", () => {
      const attr = attrInts("shape", [1, 2, 3]);
      expect(attr.name).toBe("shape");
      expect(attr.intsValue).toEqual([1n, 2n, 3n]);
      expect(attr.type).toBe("INTS");
    });

    it("should create float attribute", () => {
      const attr = attrFloat("alpha", 0.5);
      expect(attr.name).toBe("alpha");
      expect(attr.floatValue).toBe(0.5);
      expect(attr.type).toBe("FLOAT");
    });

    it("should create floats attribute", () => {
      const attr = attrFloats("values", [1.5, 2.5, 3.5]);
      expect(attr.name).toBe("values");
      expect(attr.floatsValue).toEqual([1.5, 2.5, 3.5]);
      expect(attr.type).toBe("FLOATS");
    });

    it("should create string attribute", () => {
      const attr = attrString("mode", "constant");
      expect(attr.name).toBe("mode");
      expect(attr.stringValue).toBe("constant");
      expect(attr.type).toBe("STRING");
    });
  });
});
