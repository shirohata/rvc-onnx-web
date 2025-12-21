/**
 * Pickle Parser Unit Tests
 * 
 * Tests for the Python pickle protocol parser used to read .pth files.
 */

import { describe, it, expect } from "vitest";
import { Unpickler } from "../src/pickle.js";

describe("Pickle Parser", () => {
  describe("Basic Types", () => {
    it("should parse empty dict", () => {
      // PROTO 4, EMPTY_DICT, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x7d, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({});
    });

    it("should parse empty list", () => {
      // PROTO 4, EMPTY_LIST, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x5d, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(0);
    });

    it("should parse empty tuple", () => {
      // PROTO 4, EMPTY_TUPLE, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x29, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(0);
    });

    it("should parse None", () => {
      // PROTO 4, NONE, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x4e, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(null);
    });

    it("should parse True", () => {
      // PROTO 4, NEWTRUE, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x88, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(true);
    });

    it("should parse False", () => {
      // PROTO 4, NEWFALSE, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x89, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(false);
    });
  });

  describe("Integers", () => {
    it("should parse small integers with BININT1", () => {
      // PROTO 4, BININT1 42, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x4b, 0x2a, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(42);
    });

    it("should parse medium integers with BININT2", () => {
      // PROTO 4, BININT2 1000, STOP
      // 1000 = 0x03E8 -> little endian: E8 03
      const pickle = new Uint8Array([0x80, 0x04, 0x4d, 0xe8, 0x03, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(1000);
    });
  });

  describe("Floats", () => {
    it("should parse BINFLOAT", () => {
      // PROTO 4, BINFLOAT 3.14, STOP
      // 3.14 as IEEE 754 double: 0x40091EB851EB851F (big endian in pickle)
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setFloat64(0, 3.14, false); // big endian
      const floatBytes = new Uint8Array(buffer);
      
      const pickle = new Uint8Array([0x80, 0x04, 0x47, ...floatBytes, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeCloseTo(3.14, 10);
    });
  });

  describe("Strings", () => {
    it("should parse SHORT_BINUNICODE", () => {
      // PROTO 4, SHORT_BINUNICODE "abc", STOP
      const pickle = new Uint8Array([
        0x80, 0x04,           // PROTO 4
        0x8c, 0x03,           // SHORT_BINUNICODE with length 3
        0x61, 0x62, 0x63,     // "abc"
        0x2e                  // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("abc");
    });
  });

  describe("Memo", () => {
    it("should handle MEMOIZE opcode", () => {
      // PROTO 4, SHORT_BINUNICODE "test", MEMOIZE, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x04,                 // SHORT_BINUNICODE with length 4
        0x74, 0x65, 0x73, 0x74,     // "test"
        0x94,                       // MEMOIZE
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("test");
    });
  });

  describe("Complex Structures", () => {
    it("should parse list with items", () => {
      // Build: [1, 2, 3]
      // PROTO 4, EMPTY_LIST, MEMOIZE, (BININT1 1, APPEND) x3, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,     // PROTO 4
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

    it("should parse tuple with TUPLE1", () => {
      // Build: (42,)
      // PROTO 4, BININT1 42, TUPLE1, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,     // PROTO 4
        0x4b, 0x2a,     // BININT1 42
        0x85,           // TUPLE1
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([42]);
    });

    it("should parse tuple with TUPLE2", () => {
      // Build: (1, 2)
      const pickle = new Uint8Array([
        0x80, 0x04,     // PROTO 4
        0x4b, 0x01,     // BININT1 1
        0x4b, 0x02,     // BININT1 2
        0x86,           // TUPLE2
        0x2e            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2]);
    });

    it("should parse tuple with TUPLE3", () => {
      // Build: (1, 2, 3)
      const pickle = new Uint8Array([
        0x80, 0x04,     // PROTO 4
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

  describe("Nested Structures", () => {
    it("should parse nested dict with list", () => {
      // Build: {"key": [1, 2]}
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x7d,                       // EMPTY_DICT
        0x94,                       // MEMOIZE
        0x8c, 0x03,                 // SHORT_BINUNICODE length=3
        0x6b, 0x65, 0x79,           // "key"
        0x5d,                       // EMPTY_LIST
        0x94,                       // MEMOIZE
        0x28,                       // MARK
        0x4b, 0x01,                 // BININT1 1
        0x4b, 0x02,                 // BININT1 2
        0x65,                       // APPENDS
        0x73,                       // SETITEM
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ key: [1, 2] });
    });
  });

  describe("PickleMemo (internal)", () => {
    // Note: PickleMemo is an internal class used by Unpickler,
    // not exported publicly. The tests above verify it works correctly
    // through the Unpickler interface.
    it("should support memo operations through MEMOIZE/BINGET", () => {
      // PROTO 4, SHORT_BINUNICODE "a", MEMOIZE, SHORT_BINUNICODE "b", BINGET 0, TUPLE2, STOP
      // This creates ("a", "a") by saving "a" to memo and retrieving it
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x01, 0x61,           // SHORT_BINUNICODE "a"
        0x94,                       // MEMOIZE (save "a" to memo[0])
        0x68, 0x00,                 // BINGET 0 (retrieve memo[0])
        0x86,                       // TUPLE2
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(["a", "a"]);
    });

    it("should support LONG_BINGET for large memo indices", () => {
      // Similar to BINGET but with 4-byte index
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x01, 0x78,           // SHORT_BINUNICODE "x"
        0x94,                       // MEMOIZE (save "x" to memo[0])
        0x6a, 0x00, 0x00, 0x00, 0x00, // LONG_BINGET 0 (4-byte little endian)
        0x86,                       // TUPLE2
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(["x", "x"]);
    });

    it("should support BINPUT for memo storage", () => {
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x02, 0x68, 0x69,     // SHORT_BINUNICODE "hi"
        0x71, 0x05,                 // BINPUT 5 (save to memo[5])
        0x68, 0x05,                 // BINGET 5 (retrieve memo[5])
        0x86,                       // TUPLE2
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(["hi", "hi"]);
    });

    it("should support LONG_BINPUT for large memo indices", () => {
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x01, 0x7a,           // SHORT_BINUNICODE "z"
        0x72, 0x0a, 0x00, 0x00, 0x00, // LONG_BINPUT 10 (4-byte little endian)
        0x6a, 0x0a, 0x00, 0x00, 0x00, // LONG_BINGET 10
        0x86,                       // TUPLE2
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(["z", "z"]);
    });
  });

  describe("Extended Integer Types", () => {
    it("should parse BININT with negative numbers", () => {
      // PROTO 4, BININT -1, STOP (-1 as 32-bit LE: FF FF FF FF)
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x4a, 0xff, 0xff, 0xff, 0xff, // BININT -1
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(-1);
    });

    it("should parse BININT with large positive number", () => {
      // PROTO 4, BININT 2147483647, STOP (max signed int32)
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x4a, 0xff, 0xff, 0xff, 0x7f, // BININT 2147483647
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(2147483647);
    });

    it("should parse LONG1 for variable-length integers", () => {
      // PROTO 4, LONG1 256, STOP
      // 256 requires 2 bytes: 00 01
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8a, 0x02, 0x00, 0x01,     // LONG1: length=2, bytes=00 01
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(256n); // BigInt
    });

    it("should parse LONG4 for very large integers", () => {
      // PROTO 4, LONG4 1000, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8b, 0x02, 0x00, 0x00, 0x00, // LONG4: length=2 (4-byte LE)
        0xe8, 0x03,                 // bytes for 1000
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(1000n); // BigInt
    });
  });

  describe("Bytes and Binary Data", () => {
    it("should parse SHORT_BINBYTES", () => {
      // PROTO 4, SHORT_BINBYTES [1, 2, 3], STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x43, 0x03,                 // SHORT_BINBYTES length=3
        0x01, 0x02, 0x03,           // bytes
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(new Uint8Array([1, 2, 3]));
    });

    it("should parse BINBYTES with 4-byte length", () => {
      // PROTO 4, BINBYTES [4, 5, 6], STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x42, 0x03, 0x00, 0x00, 0x00, // BINBYTES length=3 (4-byte LE)
        0x04, 0x05, 0x06,           // bytes
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(new Uint8Array([4, 5, 6]));
    });

    it("should parse BINBYTES8 with 8-byte length", () => {
      // PROTO 5, BINBYTES8 [7, 8, 9], STOP
      const pickle = new Uint8Array([
        0x80, 0x05,                 // PROTO 5
        0x8e, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // BINBYTES8 length=3 (8-byte LE)
        0x07, 0x08, 0x09,           // bytes
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual(new Uint8Array([7, 8, 9]));
    });

    it("should parse SHORT_BINSTRING", () => {
      // PROTO 2, SHORT_BINSTRING "test", STOP
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x55, 0x04,                 // SHORT_BINSTRING length=4
        0x74, 0x65, 0x73, 0x74,     // "test"
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // SHORT_BINSTRING returns string
      expect(result).toBe("test");
    });

    it("should parse BINSTRING with 4-byte length", () => {
      // PROTO 2, BINSTRING "abc", STOP
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x54, 0x03, 0x00, 0x00, 0x00, // BINSTRING length=3 (4-byte LE)
        0x61, 0x62, 0x63,           // "abc"
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("abc");
    });
  });

  describe("String Variants", () => {
    it("should parse BINUNICODE with 4-byte length", () => {
      // PROTO 4, BINUNICODE "hello", STOP
      const textEncoder = new TextEncoder();
      const helloBytes = textEncoder.encode("hello");
      
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x58, 0x05, 0x00, 0x00, 0x00, // BINUNICODE length=5 (4-byte LE)
        ...helloBytes,              // UTF-8 encoded "hello"
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("hello");
    });

    it("should parse BINUNICODE8 with 8-byte length", () => {
      // PROTO 5, BINUNICODE8 "test", STOP
      const textEncoder = new TextEncoder();
      const testBytes = textEncoder.encode("test");
      
      const pickle = new Uint8Array([
        0x80, 0x05,                 // PROTO 5
        0x8d, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // BINUNICODE8 length=4 (8-byte LE)
        ...testBytes,               // UTF-8 encoded "test"
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("test");
    });

    it("should parse empty SHORT_BINUNICODE", () => {
      // PROTO 4, SHORT_BINUNICODE "", STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x00,                 // SHORT_BINUNICODE length=0
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("");
    });
  });

  describe("Dictionary Operations", () => {
    it("should handle SETITEM for dict construction", () => {
      // Build: {"a": 1}
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x7d,                       // EMPTY_DICT
        0x94,                       // MEMOIZE
        0x8c, 0x01, 0x61,           // SHORT_BINUNICODE "a"
        0x4b, 0x01,                 // BININT1 1
        0x73,                       // SETITEM
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ a: 1 });
    });

    it("should handle SETITEMS for batch dict construction", () => {
      // Build: {"x": 10, "y": 20}
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x7d,                       // EMPTY_DICT
        0x94,                       // MEMOIZE
        0x28,                       // MARK
        0x8c, 0x01, 0x78,           // SHORT_BINUNICODE "x"
        0x4b, 0x0a,                 // BININT1 10
        0x8c, 0x01, 0x79,           // SHORT_BINUNICODE "y"
        0x4b, 0x14,                 // BININT1 20
        0x75,                       // SETITEMS
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ x: 10, y: 20 });
    });

    it("should handle empty SETITEMS", () => {
      // Build empty dict with MARK + SETITEMS
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x7d,                       // EMPTY_DICT
        0x28,                       // MARK
        0x75,                       // SETITEMS (empty)
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({});
    });
  });

  describe("List Operations", () => {
    it("should handle APPEND for list construction", () => {
      // Build: [42]
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x5d,                       // EMPTY_LIST
        0x94,                       // MEMOIZE
        0x4b, 0x2a,                 // BININT1 42
        0x61,                       // APPEND
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([42]);
    });

    it("should handle APPENDS for batch list construction", () => {
      // Build: [10, 20, 30]
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x5d,                       // EMPTY_LIST
        0x94,                       // MEMOIZE
        0x28,                       // MARK
        0x4b, 0x0a,                 // BININT1 10
        0x4b, 0x14,                 // BININT1 20
        0x4b, 0x1e,                 // BININT1 30
        0x65,                       // APPENDS
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([10, 20, 30]);
    });

    it("should handle empty APPENDS", () => {
      // Build empty list with MARK + APPENDS
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x5d,                       // EMPTY_LIST
        0x28,                       // MARK
        0x65,                       // APPENDS (empty)
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([]);
    });
  });

  describe("Tuple Operations", () => {
    it("should handle TUPLE with MARK", () => {
      // Build: (1, 2, 3, 4)
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x28,                       // MARK
        0x4b, 0x01,                 // BININT1 1
        0x4b, 0x02,                 // BININT1 2
        0x4b, 0x03,                 // BININT1 3
        0x4b, 0x04,                 // BININT1 4
        0x74,                       // TUPLE
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2, 3, 4]);
    });

    it("should handle empty TUPLE", () => {
      // Build: ()
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x28,                       // MARK
        0x74,                       // TUPLE (empty)
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([]);
    });
  });

  describe("Stack Operations", () => {
    it("should handle POP to remove item from stack", () => {
      // Push two values, pop one, create tuple with remaining
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x4b, 0x01,                 // BININT1 1
        0x4b, 0x02,                 // BININT1 2 (to be popped)
        0x30,                       // POP
        0x85,                       // TUPLE1 (only 1 remains)
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1]);
    });

    it("should handle DUP to duplicate top of stack", () => {
      // Push value, duplicate it, create tuple
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x4b, 0x05,                 // BININT1 5
        0x32,                       // DUP
        0x86,                       // TUPLE2
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([5, 5]);
    });
  });

  describe("Frame Operations", () => {
    it("should handle FRAME opcode", () => {
      // PROTO 4, FRAME, SHORT_BINUNICODE "framed", STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x95, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // FRAME length=9 (8-byte LE)
        0x8c, 0x06,                 // SHORT_BINUNICODE length=6
        0x66, 0x72, 0x61, 0x6d, 0x65, 0x64, // "framed"
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("framed");
    });
  });

  describe("Global and Build Operations", () => {
    it("should handle GLOBAL opcode for class references", () => {
      // PROTO 2, GLOBAL torch._utils _rebuild_tensor_v2, EMPTY_TUPLE, NEWOBJ, STOP
      // This simulates a basic PyTorch class instantiation
      const pickle = new Uint8Array([
        0x80, 0x02,                 // PROTO 2
        0x63,                       // GLOBAL
        ...new TextEncoder().encode("torch._utils\n_rebuild_tensor_v2\n"),
        0x29,                       // EMPTY_TUPLE
        0x81,                       // NEWOBJ
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // Should create a PythonObject with module and name
      expect(result).toHaveProperty('fullName');
      expect((result as any).fullName).toBe('torch._utils._rebuild_tensor_v2');
    });

    it("should handle STACK_GLOBAL opcode", () => {
      // PROTO 4, push module and name, STACK_GLOBAL, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x05,                 // SHORT_BINUNICODE length=5
        ...new TextEncoder().encode("torch"),
        0x8c, 0x06,                 // SHORT_BINUNICODE length=6
        ...new TextEncoder().encode("Tensor"),
        0x93,                       // STACK_GLOBAL
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toHaveProperty('fullName');
      expect((result as any).fullName).toBe('torch.Tensor');
    });

    it("should handle BUILD opcode", () => {
      // Create an object and build it with a state dict
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x04,                 // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("test"),
        0x8c, 0x03,                 // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("Obj"),
        0x93,                       // STACK_GLOBAL (creates test.Obj class)
        0x29,                       // EMPTY_TUPLE
        0x81,                       // NEWOBJ
        0x7d,                       // EMPTY_DICT (state)
        0x94,                       // MEMOIZE
        0x8c, 0x01, 0x78,           // SHORT_BINUNICODE "x"
        0x4b, 0x0a,                 // BININT1 10
        0x73,                       // SETITEM (state["x"] = 10)
        0x62,                       // BUILD (apply state to object)
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load() as any;
      expect(result).toHaveProperty('state');
      expect(result.state).toHaveProperty('x');
      expect(result.state.x).toBe(10);
    });
  });

  describe("Reduce Operations", () => {
    it("should handle REDUCE for function calls", () => {
      // PROTO 4, callable, args tuple, REDUCE, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x08,                 // SHORT_BINUNICODE length=8
        ...new TextEncoder().encode("builtins"),
        0x8c, 0x03,                 // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("int"),
        0x93,                       // STACK_GLOBAL (builtins.int)
        0x8c, 0x03,                 // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("100"),
        0x85,                       // TUPLE1 (args)
        0x52,                       // REDUCE
        0x2e                        // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // Result should be from calling the function
      expect(result).toBeDefined();
    });
  });

  describe("PERSID Operations (PyTorch Storage)", () => {
    it("should handle BINPERSID for persistent object references", () => {
      // PROTO 4, simple PERSID with tuple, BINPERSID, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                 // PROTO 4
        0x8c, 0x01, 0x30,           // SHORT_BINUNICODE "0" (storage key)
        0x85,                       // TUPLE1 (wrap in tuple for PERSID)
        0x51,                       // BINPERSID
        0x2e                        // STOP
      ]);
      const storageResolver = (key: string) => {
        expect(key).toBeDefined();
        return new Uint8Array(16); // Return storage data
      };
      const unpickler = new Unpickler(pickle, storageResolver);
      const result = unpickler.load();
      // BINPERSID returns a PersistentId object that gets resolved later
      expect(result).toBeDefined();
    });
  });
});
