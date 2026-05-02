/**
 * ONNX Protobuf Serializer
 *
 * This module serializes the OnnxModel structure to ONNX protobuf binary format.
 * It implements the ONNX wire format without requiring external dependencies.
 *
 * ONNX uses Protocol Buffers v3 format. This is a minimal implementation
 * that handles only the subset of protobuf needed for ONNX models.
 *
 * Reference:
 * - https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
 * - https://protobuf.dev/programming-guides/encoding/
 */

import {
  OnnxModel,
  OnnxGraph,
  OnnxNode,
  OnnxValueInfo,
  OnnxAttribute,
  OnnxDataType,
  TensorData,
} from "./types.js";

/**
 * Serialize an ONNX model to binary protobuf format.
 */
export function serializeOnnx(model: OnnxModel): Uint8Array {
  const writer = new ProtobufWriter();
  writeModelProto(writer, model);
  return writer.finish();
}

// =============================================================================
// Protobuf Wire Format Writer
// =============================================================================

/**
 * Protobuf wire types
 */
const enum WireType {
  Varint = 0,
  Fixed64 = 1,
  LengthDelimited = 2,
  Fixed32 = 5,
}

/**
 * Simple protobuf writer for ONNX format.
 */
class ProtobufWriter {
  private chunks: Uint8Array[] = [];
  private buffer: Uint8Array = new Uint8Array(4096);
  private pos = 0;

  /**
   * Write a varint (variable-length integer).
   * For negative numbers, protobuf encodes them as 10-byte varints 
   * representing the two's complement 64-bit representation.
   */
  writeVarint(value: number | bigint): void {
    let v = BigInt(value);
    
    // Handle negative numbers by converting to unsigned 64-bit
    if (v < 0n) {
      // Two's complement for 64-bit: add 2^64
      v = v + (1n << 64n);
    }
    
    while (v > 0x7fn) {
      this.writeByte(Number(v & 0x7fn) | 0x80);
      v >>= 7n;
    }
    this.writeByte(Number(v));
  }

  /**
   * Write a signed varint (zigzag encoded).
   */
  writeSignedVarint(value: number | bigint): void {
    const v = BigInt(value);
    // Zigzag encode: (v << 1) ^ (v >> 63)
    const encoded = (v << 1n) ^ (v >> 63n);
    this.writeVarint(encoded);
  }

  /**
   * Write a field tag (field number + wire type).
   */
  writeTag(fieldNumber: number, wireType: WireType): void {
    this.writeVarint((fieldNumber << 3) | wireType);
  }

  /**
   * Write a single byte.
   */
  writeByte(value: number): void {
    this.ensureCapacity(1);
    this.buffer[this.pos++] = value;
  }

  /**
   * Write raw bytes.
   */
  writeBytes(data: Uint8Array): void {
    this.ensureCapacity(data.length);
    this.buffer.set(data, this.pos);
    this.pos += data.length;
  }

  /**
   * Write a fixed 32-bit value (little-endian).
   */
  writeFixed32(value: number): void {
    this.ensureCapacity(4);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset);
    view.setUint32(this.pos, value, true);
    this.pos += 4;
  }

  /**
   * Write a fixed 64-bit value (little-endian).
   */
  writeFixed64(value: bigint): void {
    this.ensureCapacity(8);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset);
    view.setBigUint64(this.pos, value, true);
    this.pos += 8;
  }

  /**
   * Write a float (32-bit).
   */
  writeFloat(value: number): void {
    this.ensureCapacity(4);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset);
    view.setFloat32(this.pos, value, true);
    this.pos += 4;
  }

  /**
   * Write a double (64-bit).
   */
  writeDouble(value: number): void {
    this.ensureCapacity(8);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset);
    view.setFloat64(this.pos, value, true);
    this.pos += 8;
  }

  /**
   * Write a length-delimited string.
   */
  writeString(fieldNumber: number, value: string): void {
    const encoded = new TextEncoder().encode(value);
    this.writeTag(fieldNumber, WireType.LengthDelimited);
    this.writeVarint(encoded.length);
    this.writeBytes(encoded);
  }

  /**
   * Write a length-delimited bytes field.
   */
  writeBytesField(fieldNumber: number, value: Uint8Array): void {
    this.writeTag(fieldNumber, WireType.LengthDelimited);
    this.writeVarint(value.length);
    this.writeBytes(value);
  }

  /**
   * Write a varint field.
   */
  writeVarintField(fieldNumber: number, value: number | bigint): void {
    this.writeTag(fieldNumber, WireType.Varint);
    this.writeVarint(value);
  }

  /**
   * Write a fixed64 field.
   */
  writeFixed64Field(fieldNumber: number, value: bigint): void {
    this.writeTag(fieldNumber, WireType.Fixed64);
    this.writeFixed64(value);
  }

  /**
   * Write a float field.
   */
  writeFloatField(fieldNumber: number, value: number): void {
    this.writeTag(fieldNumber, WireType.Fixed32);
    this.writeFloat(value);
  }

  /**
   * Write a double field.
   */
  writeDoubleField(fieldNumber: number, value: number): void {
    this.writeTag(fieldNumber, WireType.Fixed64);
    this.writeDouble(value);
  }

  /**
   * Write an embedded message.
   */
  writeMessage(
    fieldNumber: number,
    writeFn: (writer: ProtobufWriter) => void
  ): void {
    const subWriter = new ProtobufWriter();
    writeFn(subWriter);
    const data = subWriter.finish();
    this.writeBytesField(fieldNumber, data);
  }

  /**
   * Write packed repeated int64 field.
   */
  writePackedInt64(fieldNumber: number, values: bigint[]): void {
    if (values.length === 0) return;
    const subWriter = new ProtobufWriter();
    for (const v of values) {
      subWriter.writeVarint(v);
    }
    this.writeBytesField(fieldNumber, subWriter.finish());
  }

  /**
   * Write packed repeated float field.
   */
  writePackedFloat(fieldNumber: number, values: number[] | Float32Array): void {
    if (values.length === 0) return;
    const data = new Uint8Array(values.length * 4);
    const view = new DataView(data.buffer);
    for (let i = 0; i < values.length; i++) {
      view.setFloat32(i * 4, values[i], true);
    }
    this.writeBytesField(fieldNumber, data);
  }

  /**
   * Ensure we have enough buffer space.
   */
  private ensureCapacity(needed: number): void {
    if (this.pos + needed > this.buffer.length) {
      // Flush current buffer to chunks
      this.chunks.push(this.buffer.slice(0, this.pos));
      // Allocate new buffer
      this.buffer = new Uint8Array(Math.max(4096, needed));
      this.pos = 0;
    }
  }

  /**
   * Finish writing and return the complete buffer.
   */
  finish(): Uint8Array {
    // Flush remaining data
    if (this.pos > 0) {
      this.chunks.push(this.buffer.slice(0, this.pos));
    }

    // Calculate total size
    const totalSize = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);

    // Combine all chunks
    const result = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }
}

// =============================================================================
// ONNX Proto Writers
// =============================================================================

/**
 * ONNX ModelProto field numbers (from onnx.proto)
 */
const ModelProtoFields = {
  ir_version: 1,
  opset_import: 8,
  producer_name: 2,
  producer_version: 3,
  domain: 4,
  model_version: 5,
  doc_string: 6,
  graph: 7,
};

/**
 * Write ModelProto
 */
function writeModelProto(writer: ProtobufWriter, model: OnnxModel): void {
  // ir_version
  writer.writeVarintField(ModelProtoFields.ir_version, model.irVersion);

  // producer_name
  if (model.producerName) {
    writer.writeString(ModelProtoFields.producer_name, model.producerName);
  }

  // producer_version
  if (model.producerVersion) {
    writer.writeString(ModelProtoFields.producer_version, model.producerVersion);
  }

  // opset_import (repeated)
  for (const opset of model.opsetImports) {
    writer.writeMessage(ModelProtoFields.opset_import, (w) => {
      if (opset.domain) {
        w.writeString(1, opset.domain); // domain
      }
      w.writeVarintField(2, opset.version); // version
    });
  }

  // graph
  writer.writeMessage(ModelProtoFields.graph, (w) => writeGraphProto(w, model.graph));
}

/**
 * ONNX GraphProto field numbers
 */
const GraphProtoFields = {
  node: 1,
  name: 2,
  initializer: 5,
  doc_string: 10,
  input: 11,
  output: 12,
  value_info: 13,
};

/**
 * Write GraphProto
 */
function writeGraphProto(writer: ProtobufWriter, graph: OnnxGraph): void {
  // name
  if (graph.name) {
    writer.writeString(GraphProtoFields.name, graph.name);
  }

  // nodes (repeated)
  for (const node of graph.nodes) {
    writer.writeMessage(GraphProtoFields.node, (w) => writeNodeProto(w, node));
  }

  // inputs (repeated)
  for (const input of graph.inputs) {
    writer.writeMessage(GraphProtoFields.input, (w) => writeValueInfoProto(w, input));
  }

  // outputs (repeated)
  for (const output of graph.outputs) {
    writer.writeMessage(GraphProtoFields.output, (w) => writeValueInfoProto(w, output));
  }

  // initializers (repeated)
  for (const init of graph.initializers) {
    writer.writeMessage(GraphProtoFields.initializer, (w) =>
      writeTensorProto(w, init.name, init.data)
    );
  }
}

/**
 * ONNX NodeProto field numbers
 */
const NodeProtoFields = {
  input: 1,
  output: 2,
  name: 3,
  op_type: 4,
  domain: 7,
  attribute: 5,
  doc_string: 6,
};

/**
 * Write NodeProto
 */
function writeNodeProto(writer: ProtobufWriter, node: OnnxNode): void {
  // inputs (repeated)
  for (const input of node.inputs) {
    writer.writeString(NodeProtoFields.input, input);
  }

  // outputs (repeated)
  for (const output of node.outputs) {
    writer.writeString(NodeProtoFields.output, output);
  }

  // name
  if (node.name) {
    writer.writeString(NodeProtoFields.name, node.name);
  }

  // op_type
  writer.writeString(NodeProtoFields.op_type, node.opType);

  // attributes (repeated)
  for (const attr of node.attributes) {
    writer.writeMessage(NodeProtoFields.attribute, (w) => writeAttributeProto(w, attr));
  }
}

/**
 * ONNX AttributeProto field numbers and types
 */
const AttributeProtoFields = {
  name: 1,
  ref_attr_name: 21,
  doc_string: 13,
  type: 20,
  f: 2, // float
  i: 3, // int64
  s: 4, // bytes
  t: 5, // TensorProto
  g: 6, // GraphProto
  floats: 7, // repeated float
  ints: 8, // repeated int64
  strings: 9, // repeated bytes
  tensors: 10,
  graphs: 11,
};

const AttributeType = {
  UNDEFINED: 0,
  FLOAT: 1,
  INT: 2,
  STRING: 3,
  TENSOR: 4,
  GRAPH: 5,
  FLOATS: 6,
  INTS: 7,
  STRINGS: 8,
  TENSORS: 9,
  GRAPHS: 10,
};

/**
 * Write AttributeProto
 */
function writeAttributeProto(writer: ProtobufWriter, attr: OnnxAttribute): void {
  // name
  writer.writeString(AttributeProtoFields.name, attr.name);

  switch (attr.type) {
    case "INT":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.INT);
      writer.writeVarintField(AttributeProtoFields.i, attr.intValue!);
      break;

    case "INTS":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.INTS);
      for (const v of attr.intsValue!) {
        writer.writeVarintField(AttributeProtoFields.ints, v);
      }
      break;

    case "FLOAT":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.FLOAT);
      writer.writeFloatField(AttributeProtoFields.f, attr.floatValue!);
      break;

    case "FLOATS":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.FLOATS);
      for (const v of attr.floatsValue!) {
        writer.writeFloatField(AttributeProtoFields.floats, v);
      }
      break;

    case "STRING":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.STRING);
      writer.writeBytesField(
        AttributeProtoFields.s,
        new TextEncoder().encode(attr.stringValue!)
      );
      break;

    case "TENSOR":
      writer.writeVarintField(AttributeProtoFields.type, AttributeType.TENSOR);
      writer.writeMessage(AttributeProtoFields.t, (w) =>
        writeTensorProto(w, attr.name, attr.tensorValue!)
      );
      break;
  }
}

/**
 * ONNX ValueInfoProto field numbers
 */
const ValueInfoProtoFields = {
  name: 1,
  type: 2,
  doc_string: 3,
};

/**
 * Write ValueInfoProto
 */
function writeValueInfoProto(writer: ProtobufWriter, info: OnnxValueInfo): void {
  // name
  writer.writeString(ValueInfoProtoFields.name, info.name);

  // type (TypeProto)
  writer.writeMessage(ValueInfoProtoFields.type, (w) => {
    // tensor_type field (field 1 in TypeProto)
    w.writeMessage(1, (tw) => {
      // elem_type
      tw.writeVarintField(1, info.elemType);

      // shape (TensorShapeProto, field 2)
      tw.writeMessage(2, (sw) => {
        for (const dim of info.shape) {
          // dim (repeated Dimension, field 1)
          sw.writeMessage(1, (dw) => {
            if (dim.dimValue !== undefined) {
              dw.writeVarintField(1, dim.dimValue); // dim_value
            } else if (dim.dimParam !== undefined) {
              dw.writeString(2, dim.dimParam); // dim_param
            }
          });
        }
      });
    });
  });
}

/**
 * ONNX TensorProto field numbers
 */
const TensorProtoFields = {
  dims: 1, // repeated int64
  data_type: 2,
  segment: 3,
  float_data: 4, // repeated float (packed)
  int32_data: 5, // repeated int32 (packed)
  string_data: 6,
  int64_data: 7, // repeated int64 (packed)
  name: 8,
  doc_string: 12,
  raw_data: 9, // bytes
  double_data: 10,
  uint64_data: 11,
};

/**
 * Write TensorProto
 */
function writeTensorProto(
  writer: ProtobufWriter,
  name: string,
  tensor: TensorData
): void {
  // dims (repeated int64)
  for (const dim of tensor.shape) {
    writer.writeVarintField(TensorProtoFields.dims, dim);
  }

  // data_type
  const dataType = dtypeToOnnxType(tensor.dtype);
  writer.writeVarintField(TensorProtoFields.data_type, dataType);

  // name
  writer.writeString(TensorProtoFields.name, name);

  // raw_data (most efficient representation)
  const rawData = tensorDataToBytes(tensor);
  writer.writeBytesField(TensorProtoFields.raw_data, rawData);
}

/**
 * Convert dtype string to ONNX TensorProto.DataType
 */
function dtypeToOnnxType(dtype: TensorData["dtype"]): number {
  switch (dtype) {
    case "float32":
      return OnnxDataType.FLOAT;
    case "float64":
      return OnnxDataType.DOUBLE;
    case "int32":
      return OnnxDataType.INT32;
    case "int64":
      return OnnxDataType.INT64;
    case "uint8":
      return OnnxDataType.UINT8;
    default:
      return OnnxDataType.FLOAT;
  }
}

/**
 * Convert tensor data to raw bytes.
 */
function tensorDataToBytes(tensor: TensorData): Uint8Array {
  const data = tensor.data;

  if (data instanceof Uint8Array) {
    return data;
  }

  if (data instanceof Float32Array) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }

  if (data instanceof Int32Array) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }

  // Handle BigInt64Array or other types
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(
      (data as ArrayBufferView).buffer,
      (data as ArrayBufferView).byteOffset,
      (data as ArrayBufferView).byteLength
    );
  }

  throw new Error(`Unsupported tensor data type: ${typeof data}`);
}
