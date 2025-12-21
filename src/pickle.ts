/**
 * Python Pickle protocol parser for PyTorch checkpoints.
 *
 * This implements the COMPLETE Python pickle protocol (versions 0-5) to parse
 * PyTorch checkpoint files. It handles ALL opcodes and the specific object types
 * used by PyTorch's serialization.
 *
 * References:
 * - https://docs.python.org/3/library/pickle.html
 * - https://github.com/python/cpython/blob/main/Lib/pickletools.py
 * - https://peps.python.org/pep-0574/ (Protocol 5)
 */

/**
 * All supported data types (comprehensive list).
 */
export type DType =
  | "float32"
  | "float64"
  | "float16"
  | "bfloat16"
  | "int64"
  | "int32"
  | "int16"
  | "int8"
  | "uint8"
  | "bool"
  | "complex64"
  | "complex128"
  | "qint8"
  | "quint8"
  | "qint32";

/**
 * Represents a PyTorch tensor storage extracted from a pickle.
 */
export interface TorchStorage {
  /** The raw tensor data */
  data: Float32Array | Int32Array | Uint8Array | Int16Array | BigInt64Array;
  /** Tensor shape */
  shape: number[];
  /** Data type */
  dtype: DType;
  /** Stride information */
  stride?: number[];
  /** Storage offset */
  offset?: number;
  /** Whether tensor requires gradient */
  requiresGrad?: boolean;
}

/**
 * COMPLETE pickle protocol opcodes (protocols 0-5).
 */
const enum PickleOpcode {
  // =========================================================================
  // Protocol 0 - Text-based (legacy)
  // =========================================================================
  MARK = 0x28, // '(' - Push mark onto stack
  STOP = 0x2e, // '.' - Stop unpickling
  POP = 0x30, // '0' - Pop top of stack
  POP_MARK = 0x31, // '1' - Pop all items since mark
  DUP = 0x32, // '2' - Duplicate top of stack
  FLOAT = 0x46, // 'F' - Float (text)
  INT = 0x49, // 'I' - Integer (text) or bool
  LONG = 0x4c, // 'L' - Long integer (text)
  NONE = 0x4e, // 'N' - None
  PERSID = 0x50, // 'P' - Persistent id (text)
  REDUCE = 0x52, // 'R' - Call __reduce__
  STRING = 0x53, // 'S' - String (text, quoted)
  UNICODE = 0x56, // 'V' - Unicode string (text, escaped)
  APPEND = 0x61, // 'a' - Append to list
  BUILD = 0x62, // 'b' - Build object (__setstate__)
  GLOBAL = 0x63, // 'c' - Push global
  DICT = 0x64, // 'd' - Build dict from pairs
  EMPTY_DICT = 0x7d, // '}' - Push empty dict
  APPENDS = 0x65, // 'e' - Append multiple to list
  GET = 0x67, // 'g' - Get from memo (text key)
  BINGET = 0x68, // 'h' - Get from memo (1-byte index)
  INST = 0x69, // 'i' - Build instance (deprecated)
  LONG_BINGET = 0x6a, // 'j' - Get from memo (4-byte index)
  LIST = 0x6c, // 'l' - Build list from stack
  EMPTY_LIST = 0x5d, // ']' - Push empty list
  OBJ = 0x6f, // 'o' - Build object (deprecated)
  PUT = 0x70, // 'p' - Store in memo (text key)
  BINPUT = 0x71, // 'q' - Store in memo (1-byte index)
  LONG_BINPUT = 0x72, // 'r' - Store in memo (4-byte index)
  SETITEM = 0x73, // 's' - Add key/value to dict
  TUPLE = 0x74, // 't' - Build tuple from stack
  EMPTY_TUPLE = 0x29, // ')' - Push empty tuple
  SETITEMS = 0x75, // 'u' - Add key/value pairs to dict

  // =========================================================================
  // Protocol 1 - Binary
  // =========================================================================
  BININT = 0x4a, // 'J' - 4-byte signed int (little-endian)
  BININT1 = 0x4b, // 'K' - 1-byte unsigned int
  BININT2 = 0x4d, // 'M' - 2-byte unsigned int (little-endian)
  BINPERSID = 0x51, // 'Q' - Persistent id (binary)
  BINSTRING = 0x54, // 'T' - String (4-byte length)
  SHORT_BINSTRING = 0x55, // 'U' - String (1-byte length)
  BINUNICODE = 0x58, // 'X' - Unicode (4-byte length)
  BINFLOAT = 0x47, // 'G' - 8-byte float (big-endian)

  // =========================================================================
  // Protocol 2
  // =========================================================================
  PROTO = 0x80, // Protocol version marker
  NEWOBJ = 0x81, // Build object via __new__
  EXT1 = 0x82, // Extension code (1-byte)
  EXT2 = 0x83, // Extension code (2-byte)
  EXT4 = 0x84, // Extension code (4-byte)
  TUPLE1 = 0x85, // Build 1-tuple from top item
  TUPLE2 = 0x86, // Build 2-tuple from top 2 items
  TUPLE3 = 0x87, // Build 3-tuple from top 3 items
  NEWTRUE = 0x88, // Push True
  NEWFALSE = 0x89, // Push False
  LONG1 = 0x8a, // Long with 1-byte length
  LONG4 = 0x8b, // Long with 4-byte length

  // =========================================================================
  // Protocol 3
  // =========================================================================
  BINBYTES = 0x42, // 'B' - Bytes (4-byte length)
  SHORT_BINBYTES = 0x43, // 'C' - Bytes (1-byte length)

  // =========================================================================
  // Protocol 4
  // =========================================================================
  SHORT_BINUNICODE = 0x8c, // Unicode (1-byte length)
  BINUNICODE8 = 0x8d, // Unicode (8-byte length)
  BINBYTES8 = 0x8e, // Bytes (8-byte length)
  EMPTY_SET = 0x8f, // Push empty set
  ADDITEMS = 0x90, // Add items to set
  FROZENSET = 0x91, // Build frozenset from mark
  NEWOBJ_EX = 0x92, // Like NEWOBJ but with kwargs
  STACK_GLOBAL = 0x93, // Push global from stack
  MEMOIZE = 0x94, // Store top in memo (auto-index)
  FRAME = 0x95, // Frame (for chunking)

  // =========================================================================
  // Protocol 5 (PEP 574 - out-of-band data)
  // =========================================================================
  BYTEARRAY8 = 0x96, // Bytearray (8-byte length)
  NEXT_BUFFER = 0x97, // Get next out-of-band buffer
  READONLY_BUFFER = 0x98, // Make buffer read-only
}

// Map of dtype strings to TypedArray byte sizes
const DTYPE_SIZES: Record<string, number> = {
  float32: 4,
  float64: 8,
  int32: 4,
  int64: 8,
  uint8: 1,
  int8: 1,
  int16: 2,
  float16: 2,
  bfloat16: 2,
  bool: 1,
  complex64: 8,
  complex128: 16,
  qint8: 1,
  quint8: 1,
  qint32: 4,
};

/**
 * Numpy dtype mapping.
 */
const NUMPY_DTYPE_MAP: Record<string, DType> = {
  float32: "float32",
  float64: "float64",
  float16: "float16",
  int64: "int64",
  int32: "int32",
  int16: "int16",
  int8: "int8",
  uint8: "uint8",
  bool: "bool",
  "<f4": "float32",
  "<f8": "float64",
  "<f2": "float16",
  "<i8": "int64",
  "<i4": "int32",
  "<i2": "int16",
  "<i1": "int8",
  "|u1": "uint8",
  "|b1": "bool",
  ">f4": "float32",
  ">f8": "float64",
};

/**
 * Extension registry for EXT opcodes (can be extended as needed).
 */
const EXTENSION_REGISTRY: Record<number, [string, string]> = {
  // Standard extensions can be added here
  // code -> [module, name]
};

/**
 * A class placeholder for Python objects we don't fully reconstruct.
 */
export class PythonObject {
  public state: unknown = null;
  public kwargs: Record<string, unknown> = {};

  constructor(
    public readonly module: string,
    public readonly name: string,
    public readonly args: unknown[] = []
  ) {}

  get fullName(): string {
    return `${this.module}.${this.name}`;
  }
}

/**
 * Represents a persistent ID reference (used for tensor storage).
 */
class PersistentId {
  constructor(public readonly id: unknown) {}
}

/**
 * Complete Unpickler class that parses ALL Python pickle protocols (0-5).
 */
export class Unpickler {
  private pos = 0;
  private data: Uint8Array;
  private dataView: DataView;
  private stack: unknown[] = [];
  private memo: Map<number | string, unknown> = new Map();
  private markStack: number[] = [];
  private storageResolver: (key: string) => Uint8Array;
  private buffers: Uint8Array[] = [];
  private bufferIndex = 0;
  private protocol = 0;

  constructor(
    data: Uint8Array,
    storageResolver: (key: string) => Uint8Array,
    buffers: Uint8Array[] = []
  ) {
    this.data = data;
    this.dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    this.storageResolver = storageResolver;
    this.buffers = buffers;
  }

  /**
   * Load and return the pickled object.
   */
  load(): Record<string, unknown> {
    while (this.pos < this.data.length) {
      const opcode = this.data[this.pos++];

      switch (opcode) {
        // =================================================================
        // Protocol markers
        // =================================================================
        case PickleOpcode.PROTO:
          this.protocol = this.data[this.pos++];
          break;

        case PickleOpcode.FRAME:
          // Skip frame size (8 bytes) - we process the whole stream
          this.pos += 8;
          break;

        case PickleOpcode.STOP:
          return this.stack.pop() as Record<string, unknown>;

        // =================================================================
        // Stack manipulation
        // =================================================================
        case PickleOpcode.MARK:
          this.markStack.push(this.stack.length);
          break;

        case PickleOpcode.POP:
          this.stack.pop();
          break;

        case PickleOpcode.POP_MARK:
          this.popMark();
          break;

        case PickleOpcode.DUP:
          this.stack.push(this.stack[this.stack.length - 1]);
          break;

        // =================================================================
        // Singletons
        // =================================================================
        case PickleOpcode.NONE:
          this.stack.push(null);
          break;

        case PickleOpcode.NEWTRUE:
          this.stack.push(true);
          break;

        case PickleOpcode.NEWFALSE:
          this.stack.push(false);
          break;

        // =================================================================
        // Integers (text-based - Protocol 0)
        // =================================================================
        case PickleOpcode.INT: {
          const line = this.readLine();
          if (line === "00") {
            this.stack.push(false);
          } else if (line === "01") {
            this.stack.push(true);
          } else {
            this.stack.push(parseInt(line, 10));
          }
          break;
        }

        case PickleOpcode.LONG: {
          const line = this.readLine();
          // Remove trailing 'L' if present
          const numStr = line.endsWith("L") ? line.slice(0, -1) : line;
          this.stack.push(BigInt(numStr));
          break;
        }

        // =================================================================
        // Integers (binary)
        // =================================================================
        case PickleOpcode.BININT:
          this.stack.push(this.readInt32());
          break;

        case PickleOpcode.BININT1:
          this.stack.push(this.data[this.pos++]);
          break;

        case PickleOpcode.BININT2:
          this.stack.push(this.readUint16());
          break;

        case PickleOpcode.LONG1: {
          const n = this.data[this.pos++];
          this.stack.push(this.readLongBytes(n));
          break;
        }

        case PickleOpcode.LONG4: {
          const n = this.readInt32();
          this.stack.push(this.readLongBytes(n));
          break;
        }

        // =================================================================
        // Floats
        // =================================================================
        case PickleOpcode.FLOAT: {
          const line = this.readLine();
          this.stack.push(parseFloat(line));
          break;
        }

        case PickleOpcode.BINFLOAT:
          this.stack.push(this.readFloat64BE());
          break;

        // =================================================================
        // Strings (text-based - Protocol 0)
        // =================================================================
        case PickleOpcode.STRING: {
          const line = this.readLine();
          this.stack.push(this.unescapeString(line));
          break;
        }

        case PickleOpcode.UNICODE: {
          const line = this.readLine();
          this.stack.push(this.decodeUnicodeEscape(line));
          break;
        }

        // =================================================================
        // Strings (binary)
        // =================================================================
        case PickleOpcode.SHORT_BINSTRING: {
          const len = this.data[this.pos++];
          this.stack.push(this.readLatin1(len));
          break;
        }

        case PickleOpcode.BINSTRING: {
          const len = this.readInt32();
          this.stack.push(this.readLatin1(len));
          break;
        }

        case PickleOpcode.SHORT_BINUNICODE: {
          const len = this.data[this.pos++];
          this.stack.push(this.readUtf8(len));
          break;
        }

        case PickleOpcode.BINUNICODE: {
          const len = this.readUint32();
          this.stack.push(this.readUtf8(len));
          break;
        }

        case PickleOpcode.BINUNICODE8: {
          const len = Number(this.readUint64());
          this.stack.push(this.readUtf8(len));
          break;
        }

        // =================================================================
        // Bytes
        // =================================================================
        case PickleOpcode.SHORT_BINBYTES: {
          const len = this.data[this.pos++];
          this.stack.push(this.readBytesRaw(len));
          break;
        }

        case PickleOpcode.BINBYTES: {
          const len = this.readUint32();
          this.stack.push(this.readBytesRaw(len));
          break;
        }

        case PickleOpcode.BINBYTES8: {
          const len = Number(this.readUint64());
          this.stack.push(this.readBytesRaw(len));
          break;
        }

        case PickleOpcode.BYTEARRAY8: {
          const len = Number(this.readUint64());
          // Bytearray is mutable, but we treat it as Uint8Array
          this.stack.push(this.readBytesRaw(len));
          break;
        }

        // =================================================================
        // Collections - Empty
        // =================================================================
        case PickleOpcode.EMPTY_LIST:
          this.stack.push([]);
          break;

        case PickleOpcode.EMPTY_TUPLE:
          this.stack.push([]);
          break;

        case PickleOpcode.EMPTY_DICT:
          this.stack.push({});
          break;

        case PickleOpcode.EMPTY_SET:
          this.stack.push(new Set());
          break;

        // =================================================================
        // Collections - Build from mark
        // =================================================================
        case PickleOpcode.LIST: {
          const items = this.popMark();
          this.stack.push(items);
          break;
        }

        case PickleOpcode.TUPLE: {
          const items = this.popMark();
          this.stack.push(items);
          break;
        }

        case PickleOpcode.TUPLE1: {
          const a = this.stack.pop();
          this.stack.push([a]);
          break;
        }

        case PickleOpcode.TUPLE2: {
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b]);
          break;
        }

        case PickleOpcode.TUPLE3: {
          const c = this.stack.pop();
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b, c]);
          break;
        }

        case PickleOpcode.DICT: {
          const items = this.popMark();
          const dict: Record<string, unknown> = {};
          for (let i = 0; i < items.length; i += 2) {
            dict[String(items[i])] = items[i + 1];
          }
          this.stack.push(dict);
          break;
        }

        case PickleOpcode.FROZENSET: {
          const items = this.popMark();
          this.stack.push(new Set(items));
          break;
        }

        // =================================================================
        // Collection mutation
        // =================================================================
        case PickleOpcode.SETITEM: {
          const value = this.stack.pop();
          const key = this.stack.pop();
          const dict = this.stack[this.stack.length - 1] as Record<string, unknown>;
          dict[String(key)] = value;
          break;
        }

        case PickleOpcode.SETITEMS: {
          const items = this.popMark();
          const dict = this.stack[this.stack.length - 1] as Record<string, unknown>;
          for (let i = 0; i < items.length; i += 2) {
            dict[String(items[i])] = items[i + 1];
          }
          break;
        }

        case PickleOpcode.APPEND: {
          const item = this.stack.pop();
          const list = this.stack[this.stack.length - 1] as unknown[];
          list.push(item);
          break;
        }

        case PickleOpcode.APPENDS: {
          const items = this.popMark();
          const list = this.stack[this.stack.length - 1] as unknown[];
          list.push(...items);
          break;
        }

        case PickleOpcode.ADDITEMS: {
          const items = this.popMark();
          const set = this.stack[this.stack.length - 1] as Set<unknown>;
          for (const item of items) {
            set.add(item);
          }
          break;
        }

        // =================================================================
        // Object construction
        // =================================================================
        case PickleOpcode.GLOBAL: {
          const module = this.readLine();
          const name = this.readLine();
          this.stack.push(new PythonObject(module, name));
          break;
        }

        case PickleOpcode.STACK_GLOBAL: {
          const name = this.stack.pop() as string;
          const module = this.stack.pop() as string;
          this.stack.push(new PythonObject(module, name));
          break;
        }

        case PickleOpcode.REDUCE: {
          const args = this.stack.pop() as unknown[];
          const callable = this.stack.pop() as PythonObject;
          this.stack.push(this.reduce(callable, args));
          break;
        }

        case PickleOpcode.NEWOBJ: {
          const args = this.stack.pop() as unknown[];
          const cls = this.stack.pop() as PythonObject;
          this.stack.push(this.newobj(cls, args, {}));
          break;
        }

        case PickleOpcode.NEWOBJ_EX: {
          const kwargs = this.stack.pop() as Record<string, unknown>;
          const args = this.stack.pop() as unknown[];
          const cls = this.stack.pop() as PythonObject;
          this.stack.push(this.newobj(cls, args, kwargs));
          break;
        }

        case PickleOpcode.INST: {
          const module = this.readLine();
          const name = this.readLine();
          const args = this.popMark();
          this.stack.push(this.newobj(new PythonObject(module, name), args, {}));
          break;
        }

        case PickleOpcode.OBJ: {
          const args = this.popMark();
          const cls = args.shift() as PythonObject;
          this.stack.push(this.newobj(cls, args, {}));
          break;
        }

        case PickleOpcode.BUILD: {
          const state = this.stack.pop();
          const obj = this.stack[this.stack.length - 1];
          this.applyBuild(obj, state);
          break;
        }

        // =================================================================
        // Extension registry
        // =================================================================
        case PickleOpcode.EXT1: {
          const code = this.data[this.pos++];
          this.stack.push(this.getExtension(code));
          break;
        }

        case PickleOpcode.EXT2: {
          const code = this.readUint16();
          this.stack.push(this.getExtension(code));
          break;
        }

        case PickleOpcode.EXT4: {
          const code = this.readInt32();
          this.stack.push(this.getExtension(code));
          break;
        }

        // =================================================================
        // Memo operations
        // =================================================================
        case PickleOpcode.PUT: {
          const key = this.readLine();
          this.memo.set(key, this.stack[this.stack.length - 1]);
          break;
        }

        case PickleOpcode.BINPUT: {
          const idx = this.data[this.pos++];
          this.memo.set(idx, this.stack[this.stack.length - 1]);
          break;
        }

        case PickleOpcode.LONG_BINPUT: {
          const idx = this.readUint32();
          this.memo.set(idx, this.stack[this.stack.length - 1]);
          break;
        }

        case PickleOpcode.MEMOIZE: {
          this.memo.set(this.memo.size, this.stack[this.stack.length - 1]);
          break;
        }

        case PickleOpcode.GET: {
          const key = this.readLine();
          this.stack.push(this.memo.get(key) ?? this.memo.get(parseInt(key, 10)));
          break;
        }

        case PickleOpcode.BINGET: {
          const idx = this.data[this.pos++];
          this.stack.push(this.memo.get(idx));
          break;
        }

        case PickleOpcode.LONG_BINGET: {
          const idx = this.readUint32();
          this.stack.push(this.memo.get(idx));
          break;
        }

        // =================================================================
        // Persistent ID (for tensor storage)
        // =================================================================
        case PickleOpcode.PERSID: {
          const line = this.readLine();
          this.stack.push(this.persistentLoad(line));
          break;
        }

        case PickleOpcode.BINPERSID: {
          const pid = this.stack.pop();
          this.stack.push(this.persistentLoad(pid));
          break;
        }

        // =================================================================
        // Protocol 5 buffer protocol
        // =================================================================
        case PickleOpcode.NEXT_BUFFER: {
          if (this.bufferIndex >= this.buffers.length) {
            throw new Error("No more buffers available for NEXT_BUFFER");
          }
          this.stack.push(this.buffers[this.bufferIndex++]);
          break;
        }

        case PickleOpcode.READONLY_BUFFER: {
          // The top of stack is already a buffer; mark it read-only (no-op in JS)
          break;
        }

        default:
          throw new Error(
            `Unknown pickle opcode: 0x${opcode.toString(16)} ` +
              `(char: ${String.fromCharCode(opcode)}) at position ${this.pos - 1}`
          );
      }
    }

    throw new Error("Unexpected end of pickle data (missing STOP opcode)");
  }

  // =========================================================================
  // Object construction handlers
  // =========================================================================

  /**
   * Handle NEWOBJ/NEWOBJ_EX opcodes.
   */
  private newobj(
    cls: PythonObject,
    args: unknown[],
    kwargs: Record<string, unknown>
  ): unknown {
    const fullName = cls.fullName;

    // PyTorch storage types
    if (fullName.includes("Storage")) {
      return this.createStorage(cls, args);
    }

    // Default: create placeholder
    const obj = new PythonObject(cls.module, cls.name, args);
    obj.kwargs = kwargs;
    return obj;
  }

  /**
   * Handle BUILD opcode.
   */
  private applyBuild(obj: unknown, state: unknown): void {
    if (obj instanceof PythonObject) {
      obj.state = state;
    } else if (typeof obj === "object" && obj !== null && state !== null) {
      if (Array.isArray(state)) {
        // __setstate__ received a tuple: (dict, slots_dict)
        const [stateDict, slotsDict] = state;
        if (stateDict && typeof stateDict === "object") {
          Object.assign(obj, stateDict);
        }
        if (slotsDict && typeof slotsDict === "object") {
          Object.assign(obj, slotsDict);
        }
      } else if (typeof state === "object") {
        Object.assign(obj, state);
      }
    }
  }

  /**
   * Create storage object from NEWOBJ.
   */
  private createStorage(cls: PythonObject, _args: unknown[]): TorchStorage {
    const dtype = this.getDtypeFromStorageType(cls);
    return {
      data: new Float32Array(0),
      shape: [],
      dtype,
    };
  }

  /**
   * Get extension from registry.
   */
  private getExtension(code: number): PythonObject {
    const entry = EXTENSION_REGISTRY[code];
    if (entry) {
      return new PythonObject(entry[0], entry[1]);
    }
    throw new Error(`Unknown extension code: ${code}`);
  }

  /**
   * Handle persistent ID loading (for tensor storage).
   */
  private persistentLoad(pid: unknown): TorchStorage | PersistentId {
    // PyTorch persistent IDs are tuples:
    // ("storage", storage_type, storage_key, location, element_count)
    // Or for newer versions:
    // ("storage", storage_type, storage_key, location, element_count, ...)
    if (!Array.isArray(pid) || pid[0] !== "storage") {
      return new PersistentId(pid);
    }

    const [, storageType, storageKey, , elementCount] = pid as [
      string,
      PythonObject,
      string,
      string,
      number
    ];

    // Determine dtype from storage type
    const dtype = this.getDtypeFromStorageType(storageType);

    // Load the raw storage data
    let rawData: Uint8Array;
    try {
      rawData = this.storageResolver(storageKey);
    } catch (e) {
      // Storage not found - return empty storage
      console.warn(`Storage not found: ${storageKey}`);
      return {
        data: new Float32Array(0),
        shape: [],
        dtype,
      };
    }

    // Create the appropriate typed array view
    const data = this.createTypedArray(dtype, rawData.buffer as ArrayBuffer, rawData.byteOffset, elementCount);

    return {
      data,
      shape: [], // Shape will be set later by _rebuild_tensor_v2
      dtype,
    };
  }

  /**
   * Get dtype string from PyTorch storage type.
   */
  private getDtypeFromStorageType(storageType: PythonObject): DType {
    const name = storageType.name.toLowerCase();

    // Quantized types
    if (name.includes("qint8")) return "qint8";
    if (name.includes("quint8")) return "quint8";
    if (name.includes("qint32")) return "qint32";

    // Complex types
    if (name.includes("complex128") || name.includes("complexdouble"))
      return "complex128";
    if (name.includes("complex64") || name.includes("complexfloat"))
      return "complex64";

    // Floating point (order matters!)
    if (name.includes("bfloat16") || name.includes("bfloat")) return "bfloat16";
    if (name.includes("half") || name.includes("float16")) return "float16";
    if (name.includes("double") || name.includes("float64")) return "float64";
    if (name.includes("float")) return "float32";

    // Integer types
    if (name.includes("long") || name.includes("int64")) return "int64";
    if (name.includes("int") && !name.includes("8") && !name.includes("16"))
      return "int32";
    if (name.includes("short") || name.includes("int16")) return "int16";
    if (name.includes("char") || name.includes("int8")) return "int8";

    // Unsigned
    if (name.includes("byte") || name.includes("uint8")) return "uint8";

    // Boolean
    if (name.includes("bool")) return "bool";

    return "float32";
  }

  /**
   * Create a TypedArray from raw buffer.
   */
  private createTypedArray(
    dtype: DType,
    buffer: ArrayBuffer,
    offset: number,
    count: number
  ): TorchStorage["data"] {
    // Ensure alignment
    const byteSize = DTYPE_SIZES[dtype] || 4;
    const byteLength = count * byteSize;

    // Create a copy if not aligned
    let alignedBuffer = buffer;
    let actualOffset = offset;

    if (offset % byteSize !== 0) {
      const slice = new Uint8Array(buffer, offset, byteLength);
      alignedBuffer = slice.buffer.slice(
        slice.byteOffset,
        slice.byteOffset + slice.byteLength
      );
      actualOffset = 0;
    }

    switch (dtype) {
      case "float32":
        return new Float32Array(alignedBuffer, actualOffset, count);

      case "float64": {
        // Convert to float32 for browser compatibility
        const f64 = new Float64Array(alignedBuffer, actualOffset, count);
        return new Float32Array(f64);
      }

      case "float16":
        // Convert float16 to float32
        return this.convertFloat16ToFloat32(
          new Uint16Array(alignedBuffer, actualOffset, count)
        );

      case "bfloat16":
        // Convert bfloat16 to float32
        return this.convertBFloat16ToFloat32(
          new Uint16Array(alignedBuffer, actualOffset, count)
        );

      case "int64":
        return new BigInt64Array(alignedBuffer, actualOffset, count);

      case "int32":
        return new Int32Array(alignedBuffer, actualOffset, count);

      case "int16":
        return new Int16Array(alignedBuffer, actualOffset, count);

      case "int8":
        return new Int8Array(alignedBuffer, actualOffset, count) as unknown as Int32Array;

      case "uint8":
      case "bool":
      case "qint8":
      case "quint8":
        return new Uint8Array(alignedBuffer, actualOffset, count);

      case "qint32":
        return new Int32Array(alignedBuffer, actualOffset, count);

      case "complex64":
        // Complex as interleaved float32 pairs
        return new Float32Array(alignedBuffer, actualOffset, count * 2);

      case "complex128": {
        // Complex as interleaved float64 pairs, convert to float32
        const c128 = new Float64Array(alignedBuffer, actualOffset, count * 2);
        return new Float32Array(c128);
      }

      default:
        return new Float32Array(alignedBuffer, actualOffset, count);
    }
  }

  /**
   * Convert float16 to float32.
   */
  private convertFloat16ToFloat32(input: Uint16Array): Float32Array {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const h = input[i];
      const sign = (h & 0x8000) >> 15;
      const exponent = (h & 0x7c00) >> 10;
      const fraction = h & 0x03ff;

      let value: number;
      if (exponent === 0) {
        if (fraction === 0) {
          value = sign ? -0 : 0;
        } else {
          // Subnormal
          value = (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
        }
      } else if (exponent === 0x1f) {
        value = fraction ? NaN : sign ? -Infinity : Infinity;
      } else {
        value = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
      }
      output[i] = value;
    }
    return output;
  }

  /**
   * Convert bfloat16 to float32.
   */
  private convertBFloat16ToFloat32(input: Uint16Array): Float32Array {
    const output = new Float32Array(input.length);
    const buffer = new ArrayBuffer(4);
    const u32 = new Uint32Array(buffer);
    const f32 = new Float32Array(buffer);

    for (let i = 0; i < input.length; i++) {
      // BFloat16 is the upper 16 bits of float32
      u32[0] = input[i] << 16;
      output[i] = f32[0];
    }
    return output;
  }

  /**
   * Handle REDUCE opcode for known PyTorch reconstructors.
   */
  private reduce(callable: PythonObject, args: unknown[]): unknown {
    const fullName = callable.fullName;

    switch (fullName) {
      // =====================================================================
      // PyTorch tensor reconstruction
      // =====================================================================
      case "torch._utils._rebuild_tensor_v2":
      case "torch._utils._rebuild_tensor_v3": {
        return this.rebuildTensorV2(args);
      }

      case "torch._utils._rebuild_parameter": {
        // args: (tensor, requires_grad, backward_hooks)
        const [tensor, requiresGrad] = args as [TorchStorage, boolean];
        if (tensor) {
          tensor.requiresGrad = requiresGrad;
        }
        return tensor;
      }

      case "torch._utils._rebuild_qtensor": {
        return this.rebuildQTensor(args);
      }

      case "torch._utils._rebuild_sparse_tensor":
      case "torch._utils._rebuild_sparse_coo_tensor": {
        return this.rebuildSparseTensor(args);
      }

      case "torch._utils._rebuild_device_tensor_v2": {
        return this.rebuildTensorV2(args);
      }

      case "torch.storage._load_from_bytes": {
        return this.loadStorageFromBytes(args);
      }

      // =====================================================================
      // Collections
      // =====================================================================
      case "collections.OrderedDict":
      case "builtins.dict":
        return {};

      case "torch.Size":
      case "builtins.tuple":
      case "builtins.list":
        return args[0] || [];

      case "builtins.set":
        return new Set(args[0] as unknown[] || []);

      case "builtins.frozenset":
        return new Set(args[0] as unknown[] || []);

      // =====================================================================
      // Numpy
      // =====================================================================
      case "numpy.core.multiarray._reconstruct":
      case "numpy._core.multiarray._reconstruct":
        return this.rebuildNumpyArray(args);

      case "numpy.dtype":
      case "numpy.core.multiarray.dtype":
        return this.rebuildNumpyDtype(args);

      // =====================================================================
      // Codecs
      // =====================================================================
      case "_codecs.encode": {
        const [text, encoding] = args as [string, string];
        if (encoding === "latin-1" || encoding === "latin1") {
          return new TextEncoder().encode(text);
        }
        return text;
      }

      // =====================================================================
      // Functools
      // =====================================================================
      case "functools.partial": {
        const [func, ...partialArgs] = args;
        return new PythonObject("functools", "partial", [func, ...partialArgs]);
      }

      // =====================================================================
      // Default: return placeholder
      // =====================================================================
      default:
        return new PythonObject(callable.module, callable.name, args);
    }
  }

  // =========================================================================
  // PyTorch-specific tensor reconstructors
  // =========================================================================

  /**
   * Rebuild tensor from _rebuild_tensor_v2 / v3.
   */
  private rebuildTensorV2(args: unknown[]): TorchStorage | null {
    // args: (storage, offset, shape, stride, requires_grad, backward_hooks, [metadata])
    const [storage, offset, shape, stride, requiresGrad] = args as [
      TorchStorage | null,
      number,
      number[],
      number[],
      boolean
    ];

    if (!storage || !storage.data) {
      return null;
    }

    const flatData = storage.data;
    const numElements = shape.reduce((a, b) => a * b, 1) || 1;

    // Apply offset if needed
    let data = flatData;
    if (offset > 0 || numElements < flatData.length) {
      data = flatData.slice(offset, offset + numElements) as typeof flatData;
    }

    return {
      data,
      shape,
      stride,
      dtype: storage.dtype,
      requiresGrad,
    };
  }

  /**
   * Rebuild quantized tensor.
   */
  private rebuildQTensor(args: unknown[]): TorchStorage | null {
    // args: (storage, offset, shape, stride, quantizer, requires_grad, backward_hooks)
    const [storage, offset, shape, stride, , requiresGrad] = args as [
      TorchStorage | null,
      number,
      number[],
      number[],
      unknown,
      boolean
    ];

    if (!storage || !storage.data) {
      return null;
    }

    const flatData = storage.data;
    const numElements = shape.reduce((a, b) => a * b, 1) || 1;

    let data = flatData;
    if (offset > 0 || numElements < flatData.length) {
      data = flatData.slice(offset, offset + numElements) as typeof flatData;
    }

    return {
      data,
      shape,
      stride,
      dtype: storage.dtype,
      requiresGrad,
    };
  }

  /**
   * Rebuild sparse tensor (returns placeholder).
   */
  private rebuildSparseTensor(args: unknown[]): unknown {
    // Sparse tensors are complex; return placeholder with original data
    return { __type__: "sparse_tensor", args };
  }

  /**
   * Load storage from bytes (used in some checkpoints).
   */
  private loadStorageFromBytes(args: unknown[]): TorchStorage {
    const [bytesData] = args as [Uint8Array];
    return {
      data: new Float32Array(
        bytesData.buffer,
        bytesData.byteOffset,
        bytesData.byteLength / 4
      ),
      shape: [],
      dtype: "float32",
    };
  }

  /**
   * Rebuild numpy array (placeholder).
   */
  private rebuildNumpyArray(args: unknown[]): unknown {
    return { __type__: "ndarray", args };
  }

  /**
   * Rebuild numpy dtype.
   */
  private rebuildNumpyDtype(args: unknown[]): DType {
    const [typeStr] = args as [string];
    return NUMPY_DTYPE_MAP[typeStr] || "float32";
  }

  // =========================================================================
  // String helpers
  // =========================================================================

  /**
   * Unescape a Python string literal.
   */
  private unescapeString(s: string): string {
    // Remove surrounding quotes
    if (
      (s.startsWith("'") && s.endsWith("'")) ||
      (s.startsWith('"') && s.endsWith('"'))
    ) {
      s = s.slice(1, -1);
    }

    // Handle escape sequences
    return s
      .replace(/\\n/g, "\n")
      .replace(/\\r/g, "\r")
      .replace(/\\t/g, "\t")
      .replace(/\\\\/g, "\\")
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\x([0-9a-fA-F]{2})/g, (_, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      );
  }

  /**
   * Decode Python unicode escape sequences.
   */
  private decodeUnicodeEscape(s: string): string {
    return s
      .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      )
      .replace(/\\U([0-9a-fA-F]{8})/g, (_, hex) =>
        String.fromCodePoint(parseInt(hex, 16))
      )
      .replace(/\\x([0-9a-fA-F]{2})/g, (_, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      )
      .replace(/\\n/g, "\n")
      .replace(/\\r/g, "\r")
      .replace(/\\t/g, "\t")
      .replace(/\\\\/g, "\\");
  }

  // =========================================================================
  // Binary reading helpers
  // =========================================================================

  private readInt32(): number {
    const val = this.dataView.getInt32(this.pos, true);
    this.pos += 4;
    return val;
  }

  private readUint16(): number {
    const val = this.dataView.getUint16(this.pos, true);
    this.pos += 2;
    return val;
  }

  private readUint32(): number {
    const val = this.dataView.getUint32(this.pos, true);
    this.pos += 4;
    return val;
  }

  private readUint64(): bigint {
    const val = this.dataView.getBigUint64(this.pos, true);
    this.pos += 8;
    return val;
  }

  private readFloat64BE(): number {
    // BINFLOAT uses big-endian!
    const val = this.dataView.getFloat64(this.pos, false);
    this.pos += 8;
    return val;
  }

  private readLongBytes(n: number): bigint {
    if (n === 0) return 0n;
    const bytes = this.data.slice(this.pos, this.pos + n);
    this.pos += n;
    let result = 0n;
    for (let i = n - 1; i >= 0; i--) {
      result = (result << 8n) | BigInt(bytes[i]);
    }
    // Handle negative numbers (two's complement)
    if (bytes[n - 1] & 0x80) {
      result -= 1n << BigInt(n * 8);
    }
    return result;
  }

  private readUtf8(len: number): string {
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    return new TextDecoder().decode(bytes);
  }

  private readLatin1(len: number): string {
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    // Latin-1 decode for pickle STRING
    return Array.from(bytes)
      .map((b) => String.fromCharCode(b))
      .join("");
  }

  private readBytesRaw(len: number): Uint8Array {
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    return bytes;
  }

  private readLine(): string {
    let end = this.pos;
    while (end < this.data.length && this.data[end] !== 0x0a) {
      end++;
    }
    const line = new TextDecoder().decode(this.data.slice(this.pos, end));
    this.pos = end + 1; // Skip the newline
    return line;
  }

  private popMark(): unknown[] {
    const markIdx = this.markStack.pop()!;
    const items = this.stack.splice(markIdx);
    return items;
  }
}
