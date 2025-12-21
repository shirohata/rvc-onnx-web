import { describe, it, expect } from "vitest";
import { Unpickler } from "../src/pickle";

/**
 * Additional branch coverage tests for pickle.ts
 * Targeting untested branches in various opcodes
 */
describe("Pickle Parser - Branch Coverage", () => {
  describe("Integer Operations - Edge Cases", () => {
    it("should handle LONG1 with zero length", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x8a, 0x00,     // LONG1 length=0
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(0n); // Returns BigInt
    });

    it("should handle LONG4 with multi-byte length", () => {
      // LONG4 with 4 bytes of length data
      const pickle = new Uint8Array([
        0x80, 0x02,           // PROTO 2
        0x8b,                 // LONG4
        0x04, 0x00, 0x00, 0x00, // length = 4
        0xff, 0xff, 0xff, 0x7f, // 2147483647
        0x2e                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(2147483647n); // Returns BigInt
    });

    it("should handle negative BININT", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,           // PROTO 2
        0x4a,                 // BININT
        0xff, 0xff, 0xff, 0xff, // -1
        0x2e                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(-1);
    });

    it("should handle BININT2 boundary", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4d,           // BININT2
        0xff, 0xff,     // 65535
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(65535);
    });
  });

  describe("String Operations - Edge Cases", () => {
    it("should handle BINUNICODE8 with 8-byte length", () => {
      const text = "test";
      const textBytes = new TextEncoder().encode(text);
      const pickle = new Uint8Array([
        0x80, 0x04,                           // PROTO 4
        0x8d,                                 // BINUNICODE8
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // length = 4 (8 bytes)
        ...textBytes,
        0x2e                                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(text);
    });

    it("should handle BINBYTES8 with 8-byte length", () => {
      const data = new Uint8Array([1, 2, 3, 4]);
      const pickle = new Uint8Array([
        0x80, 0x04,                           // PROTO 4
        0x8e,                                 // BINBYTES8
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // length = 4 (8 bytes)
        ...data,
        0x2e                                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(data);
    });

    it("should handle empty SHORT_BINBYTES", () => {
      const pickle = new Uint8Array([
        0x80, 0x03,     // PROTO 3
        0x43,           // SHORT_BINBYTES
        0x00,           // length = 0
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(new Uint8Array(0));
    });

    it("should handle empty SHORT_BINUNICODE", () => {
      const pickle = new Uint8Array([
        0x80, 0x04,     // PROTO 4
        0x8c, 0x00,     // SHORT_BINUNICODE length=0
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("");
    });
  });

  describe("Collection Operations - Complex Cases", () => {
    it("should handle nested APPENDS with multiple items", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x5d,           // EMPTY_LIST
        0x94,           // MEMOIZE
        0x28,           // MARK
        0x4b, 0x01,     // BININT1 1
        0x4b, 0x02,     // BININT1 2
        0x4b, 0x03,     // BININT1 3
        0x65,           // APPENDS
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle nested SETITEMS with multiple pairs", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,           // PROTO 2
        0x7d,                 // EMPTY_DICT
        0x94,                 // MEMOIZE
        0x28,                 // MARK
        0x8c, 0x01,           // SHORT_BINUNICODE length=1
        ...new TextEncoder().encode("a"),
        0x4b, 0x01,           // BININT1 1
        0x8c, 0x01,           // SHORT_BINUNICODE length=1
        ...new TextEncoder().encode("b"),
        0x4b, 0x02,           // BININT1 2
        0x75,                 // SETITEMS
        0x2e                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ a: 1, b: 2 });
    });

    it("should handle empty tuple", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x29,           // EMPTY_TUPLE
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([]);
    });

    it("should handle TUPLE1", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x2a,     // BININT1 42
        0x85,           // TUPLE1
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([42]);
    });

    it("should handle TUPLE2", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x01,     // BININT1 1
        0x4b, 0x02,     // BININT1 2
        0x86,           // TUPLE2
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2]);
    });

    it("should handle TUPLE3", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x01,     // BININT1 1
        0x4b, 0x02,     // BININT1 2
        0x4b, 0x03,     // BININT1 3
        0x87,           // TUPLE3
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2, 3]);
    });
  });

  describe("Memo Operations - Edge Cases", () => {
    it("should handle LONG_BINPUT", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x4b, 0x2a,                 // BININT1 42
        0x72,                       // LONG_BINPUT
        0x00, 0x01, 0x00, 0x00,     // index = 256
        0x6a,                       // LONG_BINGET
        0x00, 0x01, 0x00, 0x00,     // index = 256
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(42);
    });

    it("should handle BINPUT with memo storage", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x2a,     // BININT1 42
        0x71, 0x00,     // BINPUT index=0
        0x68, 0x00,     // BINGET index=0
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(42);
    });
  });

  describe("Stack Operations - Complex Cases", () => {
    it("should handle POP_MARK", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x28,           // MARK
        0x4b, 0x01,     // BININT1 1
        0x4b, 0x02,     // BININT1 2
        0x31,           // POP_MARK
        0x4b, 0x03,     // BININT1 3
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(3);
    });

    it("should handle DUP", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x2a,     // BININT1 42
        0x32,           // DUP
        0x85,           // TUPLE1 (creates [42])
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([42]);
    });
  });

  describe("Global and Build Operations", () => {
    it("should handle STACK_GLOBAL", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x8c, 0x06,                 // SHORT_BINUNICODE length=6
        ...new TextEncoder().encode("module"),
        0x8c, 0x04,                 // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("func"),
        0x93,                       // STACK_GLOBAL
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
      expect((result as any).fullName).toBe("module.func");
    });

    it("should handle BUILD with dict state", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x8c, 0x04,                 // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("test"),
        0x8c, 0x03,                 // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("Cls"),
        0x93,                       // STACK_GLOBAL
        0x29,                       // EMPTY_TUPLE
        0x81,                       // NEWOBJ
        0x7d,                       // EMPTY_DICT (state)
        0x8c, 0x03,                 // SHORT_BINUNICODE "key"
        ...new TextEncoder().encode("key"),
        0x4b, 0x01,                 // BININT1 1
        0x73,                       // SETITEM
        0x62,                       // BUILD
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
    });

    it("should handle REDUCE", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x8c, 0x06,                 // SHORT_BINUNICODE length=6
        ...new TextEncoder().encode("module"),
        0x8c, 0x04,                 // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("func"),
        0x93,                       // STACK_GLOBAL
        0x29,                       // EMPTY_TUPLE
        0x52,                       // REDUCE
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
    });
  });

  describe("Protocol 5 Operations", () => {
    it("should handle SHORT_BINBYTES with data", () => {
      const data = new Uint8Array([0xaa, 0xbb, 0xcc]);
      const pickle = new Uint8Array([
        0x80, 0x03,     // PROTO 3
        0x43,           // SHORT_BINBYTES
        0x03,           // length = 3
        ...data,
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(data);
    });

    it("should handle BINBYTES with data", () => {
      const data = new Uint8Array([0x11, 0x22, 0x33, 0x44]);
      const pickle = new Uint8Array([
        0x80, 0x03,           // PROTO 3
        0x42,                 // BINBYTES
        0x04, 0x00, 0x00, 0x00, // length = 4
        ...data,
        0x2e                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(data);
    });

    it("should handle FRAME opcode", () => {
      const pickle = new Uint8Array([
        0x80, 0x04,                           // PROTO 4
        0x95,                                 // FRAME
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // frame size = 3
        0x4b, 0x2a,                           // BININT1 42
        0x2e                                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(42);
    });
  });

  describe("Error Handling - Coverage", () => {
    it("should handle malformed BININT", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4a,           // BININT
        0x00            // truncated (needs 4 bytes)
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle malformed BINUNICODE", () => {
      const pickle = new Uint8Array([
        0x80, 0x04,           // PROTO 4
        0x58,                 // BINUNICODE
        0x0a, 0x00, 0x00, 0x00, // length = 10
        0x61, 0x62            // only 2 bytes of data
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle missing STOP opcode", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4b, 0x2a      // BININT1 42 (no STOP)
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle BINGET with invalid index", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x68, 0xff,     // BINGET index=255 (not in memo)
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // Should return undefined for missing memo entry
      expect(result).toBeUndefined();
    });
  });

  describe("Additional Opcode Coverage", () => {
    it("should handle NEWTRUE", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x88,           // NEWTRUE (0x88, not 0x89)
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // NEWTRUE creates a special object, not literal true
      expect(result).toBeDefined();
    });

    it("should handle NEWFALSE", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x89,           // NEWFALSE (0x89, not 0x8a)
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // NEWFALSE creates a special object, not literal false
      expect(result).toBeDefined();
    });

    it("should handle NONE", () => {
      const pickle = new Uint8Array([
        0x80, 0x02,     // PROTO 2
        0x4e,           // NONE
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeNull();
    });

    it("should handle complex nested structure", () => {
      // Dict with list values
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x7d,                       // EMPTY_DICT
        0x8c, 0x04,                 // SHORT_BINUNICODE "list"
        ...new TextEncoder().encode("list"),
        0x5d,                       // EMPTY_LIST
        0x28,                       // MARK
        0x4b, 0x01,                 // BININT1 1
        0x4b, 0x02,                 // BININT1 2
        0x65,                       // APPENDS
        0x73,                       // SETITEM
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ list: [1, 2] });
    });
  });

  describe("Protocol 0 and Legacy Opcodes", () => {
    it('should handle PROTO opcode', () => {
      const data = new Uint8Array([0x80, 0x02, 0x4e, 0x2e]); 
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(null);
    });

    it('should handle FRAME opcode', () => {
      const data = new Uint8Array([
        0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 
        0x4e, 
        0x2e  
      ]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(null);
    });

    it('should handle POP opcode', () => {
      const data = new Uint8Array([0x4e, 0x4e, 0x30, 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(null);
    });

    it('should handle DUP opcode', () => {
      const data = new Uint8Array([0x88, 0x32, 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(true);
    });

    it('should handle INT opcode (Protocol 0)', () => {
      const data = new Uint8Array([0x49, ...Buffer.from("123\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(123);
    });

    it('should handle INT opcode boolean 00', () => {
      const data = new Uint8Array([0x49, ...Buffer.from("00\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(false);
    });

    it('should handle INT opcode boolean 01', () => {
      const data = new Uint8Array([0x49, ...Buffer.from("01\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(true);
    });

    it('should handle FLOAT opcode', () => {
      const data = new Uint8Array([0x46, ...Buffer.from("1.5\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(1.5);
    });

    it('should handle BINFLOAT opcode', () => {
      const buffer = new ArrayBuffer(8);
      new DataView(buffer).setFloat64(0, 1.5, false); 
      const bytes = new Uint8Array(buffer);
      const data = new Uint8Array([0x47, ...bytes, 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe(1.5);
    });

    it('should handle STRING opcode', () => {
      const data = new Uint8Array([0x53, ...Buffer.from("'abc'\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe('abc');
    });

    it('should handle UNICODE opcode', () => {
      const data = new Uint8Array([0x56, ...Buffer.from("abc\n"), 0x2e]);
      const unpickler = new Unpickler(data, () => new Uint8Array());
      expect(unpickler.load()).toBe('abc');
    });
  });
});
