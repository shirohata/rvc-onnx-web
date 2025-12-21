/**
 * ONNX Serializer Unit Tests
 * 
 * Tests for the protobuf serialization of ONNX models.
 */

import { describe, it, expect } from "vitest";
import { serializeOnnx } from "../src/onnx-serializer.js";
import { OnnxDataType } from "../src/types.js";

describe("ONNX Serializer", () => {
  describe("Basic Serialization", () => {
    it("should serialize a minimal model", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      expect(bytes).toBeInstanceOf(Uint8Array);
      expect(bytes.length).toBeGreaterThan(0);
      // ONNX files start with field tag for ir_version (field 1, varint = 0x08)
      expect(bytes[0]).toBe(0x08);
    });

    it("should include ir_version in output", () => {
      const model = {
        irVersion: 9n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      // Field 1 (ir_version) = 0x08, followed by varint 9 = 0x09
      expect(bytes[0]).toBe(0x08);
      expect(bytes[1]).toBe(0x09);
    });
  });

  describe("Nodes", () => {
    it("should serialize model with nodes", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [{
            opType: 'Identity',
            name: 'identity_0',
            inputs: ['input'],
            outputs: ['output'],
            attributes: []
          }],
          inputs: [{
            name: 'input',
            elemType: OnnxDataType.FLOAT,
            shape: [{ dimValue: 1n }, { dimValue: 10n }]
          }],
          outputs: [{
            name: 'output',
            elemType: OnnxDataType.FLOAT,
            shape: [{ dimValue: 1n }, { dimValue: 10n }]
          }],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      expect(bytes.length).toBeGreaterThan(50);
      
      // Verify the bytes contain "Identity" somewhere
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('Identity');
    });

    it("should serialize nodes with multiple inputs/outputs", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [{
            opType: 'Add',
            name: 'add_0',
            inputs: ['a', 'b'],
            outputs: ['c'],
            attributes: []
          }],
          inputs: [
            { name: 'a', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }] },
            { name: 'b', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }] }
          ],
          outputs: [
            { name: 'c', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }] }
          ],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('Add');
      expect(str).toContain('add_0');
    });
  });

  describe("Attributes", () => {
    it("should serialize int attributes", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [{
            opType: 'Squeeze',
            name: 'squeeze_0',
            inputs: ['x'],
            outputs: ['out'],
            attributes: [{
              name: 'axes',
              type: 'INTS' as const,
              intsValue: [0n]
            }]
          }],
          inputs: [{ name: 'x', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }, { dimValue: 10n }] }],
          outputs: [{ name: 'out', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 10n }] }],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('axes');
    });

    it("should serialize float attributes", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [{
            opType: 'LeakyRelu',
            name: 'relu_0',
            inputs: ['x'],
            outputs: ['y'],
            attributes: [{
              name: 'alpha',
              type: 'FLOAT' as const,
              value: 0.01
            }]
          }],
          inputs: [{ name: 'x', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }] }],
          outputs: [{ name: 'y', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 1n }] }],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('alpha');
    });

    it("should serialize ints list attributes", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [{
            opType: 'Reshape',
            name: 'reshape_0',
            inputs: ['x', 'shape'],
            outputs: ['y'],
            attributes: []
          }],
          inputs: [{ name: 'x', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 6n }] }],
          outputs: [{ name: 'y', elemType: OnnxDataType.FLOAT, shape: [{ dimValue: 2n }, { dimValue: 3n }] }],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      expect(bytes.length).toBeGreaterThan(20);
    });
  });

  describe("Initializers", () => {
    it("should serialize float32 initializers", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: [{
            name: 'weights',
            data: {
              data: new Float32Array([1.0, 2.0, 3.0, 4.0]),
              shape: [2, 2],
              dtype: 'float32' as const
            }
          }]
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('weights');
    });

    it("should serialize int64 initializers", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: [{
            name: 'indices',
            data: {
              data: new BigInt64Array([1n, 2n, 3n]),
              shape: [3],
              dtype: 'int64' as const
            }
          }]
        }
      };

      const bytes = serializeOnnx(model);
      expect(bytes.length).toBeGreaterThan(20);
    });

    it("should serialize scalar initializers", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: [{
            name: 'scalar',
            data: {
              data: new Float32Array([3.14]),
              shape: [],
              dtype: 'float32' as const
            }
          }]
        }
      };

      const bytes = serializeOnnx(model);
      expect(bytes.length).toBeGreaterThan(10);
    });
  });

  describe("Dynamic Shapes", () => {
    it("should serialize dynamic dimensions", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [{
            name: 'input',
            elemType: OnnxDataType.FLOAT,
            shape: [
              { dimValue: 1n },           // batch = 1
              { dimParam: 'seq_len' },    // dynamic
              { dimValue: 768n }          // hidden
            ]
          }],
          outputs: [],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('seq_len');
    });
  });

  describe("Opset Imports", () => {
    it("should serialize multiple opset imports", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [
          { domain: '', version: 17n },
          { domain: 'ai.onnx.ml', version: 3n }
        ],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [],
          outputs: [],
          initializers: []
        }
      };

      const bytes = serializeOnnx(model);
      const str = new TextDecoder().decode(bytes);
      expect(str).toContain('ai.onnx.ml');
    });
  });

  describe("Data Types", () => {
    it("should handle FLOAT type", () => {
      expect(OnnxDataType.FLOAT).toBe(1);
    });

    it("should handle INT64 type", () => {
      expect(OnnxDataType.INT64).toBe(7);
    });

    it("should handle INT32 type", () => {
      expect(OnnxDataType.INT32).toBe(6);
    });

    it("should throw when serializing invalid elemType string instead of enum", () => {
      const model = {
        irVersion: 8n,
        opsetImports: [{ domain: '', version: 17n }],
        producerName: 'test',
        producerVersion: '1.0.0',
        graph: {
          name: 'test_graph',
          nodes: [],
          inputs: [{
            name: 'input',
            // Intentionally using a string instead of enum to test runtime error
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            elemType: 'FLOAT' as any,
            shape: [{ dimValue: 1n }]
          }],
          outputs: [],
          initializers: []
        }
      };

      // BigInt('FLOAT') throws SyntaxError when writeVarint tries to serialize
      expect(() => serializeOnnx(model)).toThrow();
    });
  });
});
