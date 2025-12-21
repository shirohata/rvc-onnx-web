/**
 * Extended Pickle Parser Tests
 * 
 * Additional comprehensive tests for pickle opcodes, error handling, and edge cases
 * to maximize branch and function coverage.
 */

import { describe, it, expect } from "vitest";
import { Unpickler } from "../src/pickle.js";

describe("Pickle Parser - Extended Coverage", () => {
  describe("Text Protocol Opcodes", () => {
    it("should handle INT with '01' (True)", () => {
      // Protocol 0, INT 01\n (True), STOP
      const pickle = new Uint8Array([
        0x49, 0x30, 0x31, 0x0a,  // INT "01\n"
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(true);
    });

    it("should handle INT with '00' (False)", () => {
      // Protocol 0, INT 00\n (False), STOP
      const pickle = new Uint8Array([
        0x49, 0x30, 0x30, 0x0a,  // INT "00\n"
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(false);
    });

    it("should handle INT with arbitrary number", () => {
      // Protocol 0, INT 42\n, STOP
      const pickle = new Uint8Array([
        0x49, 0x34, 0x32, 0x0a,  // INT "42\n"
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(42);
    });

    it("should handle FLOAT opcode", () => {
      // Protocol 0, FLOAT 3.14\n, STOP
      const floatStr = "3.14\n";
      const pickle = new Uint8Array([
        0x46,  // FLOAT
        ...new TextEncoder().encode(floatStr),
        0x2e   // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeCloseTo(3.14, 5);
    });

    it("should handle LONG opcode", () => {
      // Protocol 0, LONG 12345L\n, STOP
      const longStr = "12345L\n";
      const pickle = new Uint8Array([
        0x4c,  // LONG
        ...new TextEncoder().encode(longStr),
        0x2e   // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(12345n);
    });
  });

  describe("Set Operations", () => {
    it("should handle EMPTY_SET", () => {
      // Protocol 4, EMPTY_SET, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,  // PROTO 4
        0x8f,        // EMPTY_SET
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // EMPTY_SET creates an object with a Set() method, not an array
      expect(result).toBeDefined();
      expect(typeof result).toBe('object');
    });
  });

  describe("Advanced Object Construction", () => {
    it("should handle NEWOBJ_EX with kwargs", () => {
      // Protocol 4, class, args tuple, kwargs dict, NEWOBJ_EX, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,                     // PROTO 4
        0x8c, 0x04,                     // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("test"),
        0x8c, 0x03,                     // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("Cls"),
        0x93,                           // STACK_GLOBAL
        0x29,                           // EMPTY_TUPLE (args)
        0x7d,                           // EMPTY_DICT (kwargs)
        0x92,                           // NEWOBJ_EX
        0x2e                            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
      expect((result as any).fullName).toBe("test.Cls");
    });

    it("should handle NEWOBJ for object construction", () => {
      // Protocol 2, class, args, NEWOBJ, STOP
      const pickle = new Uint8Array([
        0x80, 0x02,                     // PROTO 2
        0x8c, 0x04,                     // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("test"),
        0x8c, 0x04,                     // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("Cls2"),
        0x93,                           // STACK_GLOBAL
        0x29,                           // EMPTY_TUPLE (args)
        0x81,                           // NEWOBJ
        0x2e                            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
    });
  });

  describe("Extension Registry", () => {
    it("should handle EXT1 for extension code", () => {
      // Protocol 2, EXT1 code, STOP
      const pickle = new Uint8Array([
        0x80, 0x02,  // PROTO 2
        0x82, 0x01,  // EXT1 code=1
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      // Extension codes call a registry function we don't have
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle EXT2 for extension code", () => {
      // Protocol 2, EXT2 code, STOP
      const pickle = new Uint8Array([
        0x80, 0x02,      // PROTO 2
        0x83, 0x01, 0x00, // EXT2 code=1 (2-byte LE)
        0x2e             // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle EXT4 for extension code", () => {
      // Protocol 2, EXT4 code, STOP
      const pickle = new Uint8Array([
        0x80, 0x02,              // PROTO 2
        0x84, 0x01, 0x00, 0x00, 0x00, // EXT4 code=1 (4-byte LE)
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });
  });

  describe("Memo Operations - Text Protocol", () => {
    it("should handle PUT with text key", () => {
      // Protocol 0, value, PUT 0\n, GET 0\n, TUPLE2, STOP
      const pickle = new Uint8Array([
        0x4b, 0x2a,      // BININT1 42
        0x70, 0x30, 0x0a, // PUT "0\n"
        0x67, 0x30, 0x0a, // GET "0\n"
        0x86,            // TUPLE2
        0x2e             // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([42, 42]);
    });

    it("should handle GET with text key", () => {
      // Protocol 0, value, PUT abc\n, GET abc\n, TUPLE2, STOP
      const pickle = new Uint8Array([
        0x4b, 0x05,      // BININT1 5
        0x70,            // PUT
        ...new TextEncoder().encode("abc\n"),
        0x67,            // GET
        ...new TextEncoder().encode("abc\n"),
        0x86,            // TUPLE2
        0x2e             // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([5, 5]);
    });
  });

  describe("PERSID Text Protocol", () => {
    it("should handle PERSID with text id", () => {
      // Protocol 0, PERSID storage_0\n, STOP
      const pickle = new Uint8Array([
        0x50,            // PERSID
        ...new TextEncoder().encode("storage_0\n"),
        0x2e             // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(8));
      const result = unpickler.load();
      expect(result).toBeDefined();
    });
  });

  describe("Protocol 5 Opcodes", () => {
    it("should handle BYTEARRAY8", () => {
      // Protocol 5, BYTEARRAY8 with 8-byte length, STOP
      const pickle = new Uint8Array([
        0x80, 0x05,                                         // PROTO 5
        0x96, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // BYTEARRAY8 length=4 (8-byte LE)
        0x01, 0x02, 0x03, 0x04,                            // bytes
        0x2e                                               // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeInstanceOf(Uint8Array);
      expect(result).toEqual(new Uint8Array([1, 2, 3, 4]));
    });

    it("should handle NEXT_BUFFER", () => {
      // Protocol 5, NEXT_BUFFER, STOP
      const buffer1 = new Uint8Array([10, 20, 30]);
      const pickle = new Uint8Array([
        0x80, 0x05,  // PROTO 5
        0x97,        // NEXT_BUFFER
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0), [buffer1]);
      const result = unpickler.load();
      expect(result).toEqual(buffer1);
    });

    it("should handle READONLY_BUFFER", () => {
      // Protocol 5, NEXT_BUFFER, READONLY_BUFFER, STOP
      const buffer1 = new Uint8Array([5, 15, 25]);
      const pickle = new Uint8Array([
        0x80, 0x05,  // PROTO 5
        0x97,        // NEXT_BUFFER
        0x98,        // READONLY_BUFFER (wraps previous buffer as readonly)
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0), [buffer1]);
      const result = unpickler.load();
      expect(result).toBeDefined();
    });
  });

  describe("String Variants - Text Protocol", () => {
    it("should handle STRING opcode", () => {
      // Protocol 0, STRING 'test'\n, STOP
      const strContent = "'test'\n";
      const pickle = new Uint8Array([
        0x53,  // STRING
        ...new TextEncoder().encode(strContent),
        0x2e   // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(typeof result).toBe("string");
      expect(result).toContain("test");
    });

    it("should handle UNICODE opcode", () => {
      // Protocol 0, UNICODE test\n, STOP
      const unicodeStr = "hello\\u0020world\n";
      const pickle = new Uint8Array([
        0x56,  // UNICODE
        ...new TextEncoder().encode(unicodeStr),
        0x2e   // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(typeof result).toBe("string");
    });
  });

  describe("Collection Construction", () => {
    it("should handle LIST opcode", () => {
      // Protocol 0, MARK, items, LIST, STOP
      const pickle = new Uint8Array([
        0x28,        // MARK
        0x4b, 0x01,  // BININT1 1
        0x4b, 0x02,  // BININT1 2
        0x4b, 0x03,  // BININT1 3
        0x6c,        // LIST
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle DICT opcode", () => {
      // Protocol 0, MARK, key1, val1, key2, val2, DICT, STOP
      const pickle = new Uint8Array([
        0x28,                    // MARK
        0x8c, 0x01, 0x61,        // SHORT_BINUNICODE "a"
        0x4b, 0x01,              // BININT1 1
        0x8c, 0x01, 0x62,        // SHORT_BINUNICODE "b"
        0x4b, 0x02,              // BININT1 2
        0x64,                    // DICT
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual({ a: 1, b: 2 });
    });
  });

  describe("Negative Integer Variants", () => {
    it("should handle negative BININT", () => {
      // Protocol 2, BININT -100, STOP
      const pickle = new Uint8Array([
        0x80, 0x02,                     // PROTO 2
        0x4a, 0x9c, 0xff, 0xff, 0xff,   // BININT -100 (two's complement)
        0x2e                            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(-100);
    });

    it("should handle negative LONG1", () => {
      // Protocol 4, LONG1 -256, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,              // PROTO 4
        0x8a, 0x02, 0x00, 0xff,  // LONG1: length=2, bytes represent -256
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(typeof result).toBe("bigint");
    });
  });

  describe("Error Conditions", () => {
    it("should throw on malformed BININT1", () => {
      // PROTO 4, BININT1 but truncated
      const pickle = new Uint8Array([0x80, 0x04, 0x4b]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should throw on malformed BININT2", () => {
      // PROTO 4, BININT2 but only 1 byte
      const pickle = new Uint8Array([0x80, 0x04, 0x4d, 0x01]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should throw on malformed SHORT_BINUNICODE", () => {
      // PROTO 4, SHORT_BINUNICODE length=10 but only 2 bytes
      const pickle = new Uint8Array([0x80, 0x04, 0x8c, 0x0a, 0x41, 0x42]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow();
    });

    it("should handle POP on non-empty stack", () => {
      // PROTO 4, push two values, POP one, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x4b, 0x01, 0x4b, 0x02, 0x30, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(1); // 2 was popped
    });

    it("should handle BINGET with missing memo key", () => {
      // PROTO 4, BINGET 99 (not in memo), STOP
      const pickle = new Uint8Array([0x80, 0x04, 0x68, 0x63, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      // Missing memo keys return undefined
      expect(result).toBeUndefined();
    });

    it("should throw on NEXT_BUFFER without buffer", () => {
      // PROTO 5, NEXT_BUFFER (but no buffers provided), STOP
      const pickle = new Uint8Array([0x80, 0x05, 0x97, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0), []);
      expect(() => unpickler.load()).toThrow();
    });

    it("should throw on unknown opcode", () => {
      // PROTO 4, unknown opcode 0xFF, STOP
      const pickle = new Uint8Array([0x80, 0x04, 0xff, 0x2e]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      expect(() => unpickler.load()).toThrow(/Unknown pickle opcode/);
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty BINUNICODE", () => {
      // PROTO 4, BINUNICODE length=0, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,              // PROTO 4
        0x58, 0x00, 0x00, 0x00, 0x00, // BINUNICODE length=0
        0x2e                     // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe("");
    });

    it("should handle zero-length LONG1", () => {
      // PROTO 4, LONG1 length=0 (represents 0), STOP
      const pickle = new Uint8Array([
        0x80, 0x04,  // PROTO 4
        0x8a, 0x00,  // LONG1 length=0
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(0n);
    });

    it("should handle POP_MARK correctly", () => {
      // PROTO 4, MARK, values, POP_MARK, value, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,  // PROTO 4
        0x28,        // MARK
        0x4b, 0x01,  // BININT1 1 (to be popped)
        0x4b, 0x02,  // BININT1 2 (to be popped)
        0x31,        // POP_MARK (remove mark and everything after it)
        0x4b, 0x0a,  // BININT1 10 (this remains)
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBe(10);
    });

    it("should handle multiple nested marks", () => {
      // PROTO 4, MARK, MARK, items, TUPLE, TUPLE, STOP
      const pickle = new Uint8Array([
        0x80, 0x04,  // PROTO 4
        0x28,        // MARK (outer)
        0x28,        // MARK (inner)
        0x4b, 0x01,  // BININT1 1
        0x74,        // TUPLE (builds inner tuple)
        0x74,        // TUPLE (builds outer tuple)
        0x2e         // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toEqual([[1]]);
    });

    it("should handle BUILD with non-dict state", () => {
      // Some objects use BUILD with non-dict state
      const pickle = new Uint8Array([
        0x80, 0x04,                     // PROTO 4
        0x8c, 0x04,                     // SHORT_BINUNICODE length=4
        ...new TextEncoder().encode("test"),
        0x8c, 0x03,                     // SHORT_BINUNICODE length=3
        ...new TextEncoder().encode("Obj"),
        0x93,                           // STACK_GLOBAL
        0x29,                           // EMPTY_TUPLE
        0x81,                           // NEWOBJ
        0x8c, 0x05,                     // SHORT_BINUNICODE "state" (non-dict state)
        ...new TextEncoder().encode("state"),
        0x62,                           // BUILD
        0x2e                            // STOP
      ]);
      const unpickler = new Unpickler(pickle, () => new Uint8Array(0));
      const result = unpickler.load();
      expect(result).toBeDefined();
    });
  });
});
