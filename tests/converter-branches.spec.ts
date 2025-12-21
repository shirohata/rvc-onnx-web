import { describe, it, expect } from "vitest";
import { pthToOnnx } from "../src/converter";

// Polyfill File for Node.js environment if missing
if (typeof File === 'undefined') {
  class MockFile extends Blob {
    name: string;
    lastModified: number;
    constructor(fileBits: BlobPart[], fileName: string, options?: FilePropertyBag) {
      super(fileBits, options);
      this.name = fileName;
      this.lastModified = options?.lastModified ?? Date.now();
    }
  }
  global.File = MockFile as any;
}

/**
 * Additional converter tests to cover missing branches - error paths only
 */
describe("Converter - Branch Coverage", () => {
  describe("Input Validation - Error Cases", () => {
    it("should reject null input", async () => {
      await expect(pthToOnnx(null as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject undefined input", async () => {
      await expect(pthToOnnx(undefined as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject number input", async () => {
      await expect(pthToOnnx(12345 as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject plain object input", async () => {
      await expect(pthToOnnx({ data: "test" } as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject array input", async () => {
      await expect(pthToOnnx([1, 2, 3] as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject boolean input", async () => {
      await expect(pthToOnnx(true as any)).rejects.toThrow(/Unsupported input type/);
    });

    it("should reject string input", async () => {
      await expect(pthToOnnx("invalid-string" as any)).rejects.toThrow(/String input must be a valid URL/);
    });
  });

  describe("Response Error Handling", () => {
    it("should handle Response with status 500", async () => {
      const response = new Response("Server Error", {
        status: 500,
        statusText: "Internal Server Error",
      });
      
      await expect(pthToOnnx(response)).rejects.toThrow(/Failed to fetch/);
    });

    it("should handle Response with status 403", async () => {
      const response = new Response("Forbidden", {
        status: 403,
        statusText: "Forbidden",
      });
      
      await expect(pthToOnnx(response)).rejects.toThrow(/Failed to fetch/);
    });

    it("should handle Response with empty body and error status", async () => {
      const response = new Response(null, {
        status: 404,
        statusText: "Not Found",
      });
      
      await expect(pthToOnnx(response)).rejects.toThrow(/Failed to fetch/);
    });
  });

  describe("ReadableStream Error Handling", () => {
    it("should handle ReadableStream that errors", async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.error(new Error("Stream error"));
        },
      });
      
      await expect(pthToOnnx(stream)).rejects.toThrow();
    });

    it("should handle empty ReadableStream", async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.close();
        },
      });
      
      await expect(pthToOnnx(stream)).rejects.toThrow();
    });
  });

  describe("Buffer Error Handling", () => {
    it("should handle empty ArrayBuffer", async () => {
      const emptyBuffer = new ArrayBuffer(0);
      await expect(pthToOnnx(emptyBuffer)).rejects.toThrow();
    });

    it("should handle empty Uint8Array", async () => {
      const emptyArray = new Uint8Array(0);
      await expect(pthToOnnx(emptyArray)).rejects.toThrow();
    });

    it("should handle small invalid Uint8Array", async () => {
      const smallArray = new Uint8Array([1, 2, 3]);
      await expect(pthToOnnx(smallArray)).rejects.toThrow();
    });
  });

  describe("File and Blob Error Handling", () => {
    it("should handle empty File", async () => {
      const emptyFile = new File([], "empty.pth");
      await expect(pthToOnnx(emptyFile)).rejects.toThrow();
    });

    it("should handle empty Blob", async () => {
      const emptyBlob = new Blob([]);
      await expect(pthToOnnx(emptyBlob)).rejects.toThrow();
    });

    it("should handle File with wrong content", async () => {
      const wrongContent = new File([new Uint8Array([1, 2, 3])], "wrong.pth");
      await expect(pthToOnnx(wrongContent)).rejects.toThrow();
    });
  });
});
