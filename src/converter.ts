/**
 * Main converter module - orchestrates the PTH to ONNX conversion.
 */

import { parsePth } from "./pth-parser";
import { buildOnnxModel } from "./onnx-builder";
import { serializeOnnx } from "./onnx-serializer";
import type { ParsedCheckpoint } from "./types";

/**
 * Options for the conversion process.
 */
export interface ConversionOptions {
  /** ONNX opset version (default: 17) */
  opsetVersion?: number;
  /** Sequence length for the model's phone input dimension */
  phoneLen?: number;
  /** Whether to run simplification passes */
  simplify?: boolean;
}

/**
 * Result of a successful conversion.
 */
export interface ConversionResult {
  /** The ONNX model as a binary buffer */
  onnxBuffer: Uint8Array;
  /** Parsed checkpoint metadata */
  checkpoint: ParsedCheckpoint;
  /** Model sample rate */
  sampleRate: number;
}

/**
 * Supported input types for pthToOnnx.
 * 
 * - `ArrayBuffer` - Raw binary data
 * - `Uint8Array` - Byte array (will be converted to ArrayBuffer)
 * - `File` - Browser File object (will call arrayBuffer())
 * - `Blob` - Browser Blob object (will call arrayBuffer())
 * - `Response` - Fetch Response object (will call arrayBuffer())
 * - `ReadableStream<Uint8Array>` - Stream of bytes (will be fully read)
 * - `URL` - URL object pointing to a .pth file (will be fetched)
 * - `string` - URL string pointing to a .pth file (will be fetched)
 */
export type PthInput =
  | ArrayBuffer
  | Uint8Array
  | File
  | Blob
  | Response
  | ReadableStream<Uint8Array>
  | URL
  | string;

/**
 * Normalize various input types to an ArrayBuffer.
 * 
 * @param input - The input in any supported format
 * @returns The input as an ArrayBuffer
 * @throws Error if the input type is not supported
 */
async function normalizeInput(input: PthInput): Promise<ArrayBuffer> {
  // Already an ArrayBuffer
  if (input instanceof ArrayBuffer) {
    return input;
  }

  // Uint8Array - get underlying buffer or copy
  if (input instanceof Uint8Array) {
    // If the Uint8Array is a view of a larger buffer, copy it
    if (input.byteOffset !== 0 || input.byteLength !== input.buffer.byteLength) {
      return input.slice().buffer as ArrayBuffer;
    }
    return input.buffer as ArrayBuffer;
  }

  // File or Blob - call arrayBuffer()
  if (input instanceof Blob) {
    return await input.arrayBuffer();
  }

  // Response - call arrayBuffer()
  if (input instanceof Response) {
    if (!input.ok) {
      throw new Error(`Failed to fetch: ${input.status} ${input.statusText}`);
    }
    return await input.arrayBuffer();
  }

  // ReadableStream - read all chunks
  if (typeof ReadableStream !== "undefined" && input instanceof ReadableStream) {
    const reader = input.getReader();
    const chunks: Uint8Array[] = [];
    let totalLength = 0;

    let done = false;
    while (!done) {
      const result = await reader.read();
      done = result.done;
      if (result.value) {
        chunks.push(result.value);
        totalLength += result.value.byteLength;
      }
    }

    // Combine all chunks into a single buffer
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return combined.buffer;
  }

  // URL object - fetch it
  if (typeof URL !== "undefined" && input instanceof URL) {
    const response = await fetch(input.href);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${input.href}: ${response.status} ${response.statusText}`);
    }
    return await response.arrayBuffer();
  }

  // String URL - validate and fetch
  if (typeof input === "string") {
    // Validate it looks like a URL
    if (!input.startsWith("http://") && !input.startsWith("https://") && !input.startsWith("blob:") && !input.startsWith("data:")) {
      throw new Error(
        `String input must be a valid URL (http://, https://, blob:, or data:). Got: "${input.substring(0, 50)}${input.length > 50 ? "..." : ""}"`
      );
    }
    const response = await fetch(input);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${input}: ${response.status} ${response.statusText}`);
    }
    return await response.arrayBuffer();
  }

  // Unknown type - provide helpful error
  const typeName = input === null ? "null" : typeof input === "object" ? input.constructor?.name || "Object" : typeof input;
  throw new Error(
    `Unsupported input type: ${typeName}. ` +
    `Expected ArrayBuffer, Uint8Array, File, Blob, Response, ReadableStream, URL, or string URL.`
  );
}

/**
 * Convert an RVC .pth file to ONNX format.
 *
 * This is the main entry point for browser-based conversion.
 * The entire process happens in-memory without any filesystem access.
 *
 * @param input - The .pth file in any supported format:
 *   - `ArrayBuffer` - Raw binary data
 *   - `Uint8Array` - Byte array
 *   - `File` - Browser File object (from file input)
 *   - `Blob` - Browser Blob object
 *   - `Response` - Fetch Response object
 *   - `ReadableStream<Uint8Array>` - Stream of bytes
 *   - `URL` - URL object pointing to a .pth file
 *   - `string` - URL string pointing to a .pth file
 * @param options - Conversion options
 * @returns The ONNX model as a Uint8Array
 *
 * @example
 * ```typescript
 * // From fetch response
 * const response = await fetch("model.pth");
 * const { onnxBuffer } = await pthToOnnx(response);
 * 
 * // From URL string
 * const { onnxBuffer } = await pthToOnnx("https://example.com/model.pth");
 * 
 * // From file input
 * const file = document.querySelector('input[type="file"]').files[0];
 * const { onnxBuffer } = await pthToOnnx(file);
 * 
 * // From ArrayBuffer (legacy)
 * const buffer = await fetch("model.pth").then(r => r.arrayBuffer());
 * const { onnxBuffer } = await pthToOnnx(buffer);
 * ```
 */
export async function pthToOnnx(
  input: PthInput,
  options: ConversionOptions = {}
): Promise<ConversionResult> {
  // Normalize input to ArrayBuffer
  const pthBuffer = await normalizeInput(input);
  
  const { opsetVersion = 17, phoneLen = 100, simplify = false } = options;

  // Step 1: Parse the .pth file to extract weights and config
  const checkpoint = await parsePth(pthBuffer);

  // Step 2: Build the ONNX graph structure
  const onnxModel = buildOnnxModel(checkpoint, {
    opsetVersion,
    phoneLen,
  });

  // Step 3: Serialize to binary ONNX format (protobuf)
  const onnxBuffer = serializeOnnx(onnxModel);

  // Step 4: Optionally simplify the model
  if (simplify) {
    // TODO: Implement ONNX simplification passes
    // For now, we skip this as it's complex and optional
    console.warn("ONNX simplification not yet implemented in browser");
  }

  return {
    onnxBuffer,
    checkpoint,
    sampleRate: checkpoint.config.sr,
  };
}
