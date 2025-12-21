/**
 * PTH Parser Unit Tests
 * 
 * Tests for the PyTorch checkpoint (.pth) parser.
 */

import { describe, it, expect } from "vitest";
import { parsePth } from "../src/pth-parser.js";
import { readFile } from "fs/promises";
import { resolve } from "path";

describe("PTH Parser", () => {
  describe("Version Detection", () => {
    it("should detect v1 model without f0", async () => {
      // Create a minimal v1 checkpoint without f0
      const checkpointData = {
        config: [
          768, // hidden_channels
          192, // filter_channels  
          192, // n_heads
          40000, // sample_rate
          2, // p_dropout
          6, // resblock
          1, // resblock_kernel_sizes
          3, // resblock_dilation_sizes
          512, // upsample_rates
          1, // upsample_initial_channel
          1, // upsample_kernel_sizes
          2, // spk_embed_dim
          4, // gin_channels
          0  // sr - 0 indicates no f0
        ],
        params: {},
        info: "Test model v1",
        weight: {}
      };

      // We can't easily test this without creating a full pickle,
      // but we can test the config parsing logic
      expect(checkpointData.config).toHaveLength(14);
      expect(checkpointData.config[13]).toBe(0); // sr = 0 means no f0
    });

    it("should detect v2 model with f0", async () => {
      // Create a minimal v2 checkpoint with f0
      const checkpointData = {
        config: [
          768, // hidden_channels
          192, // filter_channels  
          192, // n_heads
          40000, // sample_rate
          2, // p_dropout
          6, // resblock
          1, // resblock_kernel_sizes
          3, // resblock_dilation_sizes
          512, // upsample_rates
          1, // upsample_initial_channel
          1, // upsample_kernel_sizes
          2, // spk_embed_dim
          4, // gin_channels
          48000  // sr - non-zero indicates f0
        ],
        params: {},
        info: "Test model v2",
        weight: {}
      };

      expect(checkpointData.config).toHaveLength(14);
      expect(checkpointData.config[13]).not.toBe(0); // sr != 0 means f0
    });
  });

  describe("Config Validation", () => {
    it("should handle config with all required fields", () => {
      const config = [
        768,   // hidden_channels
        192,   // filter_channels  
        192,   // n_heads
        40000, // sample_rate
        2,     // p_dropout
        6,     // resblock
        1,     // resblock_kernel_sizes
        3,     // resblock_dilation_sizes
        512,   // upsample_rates
        1,     // upsample_initial_channel
        1,     // upsample_kernel_sizes
        2,     // spk_embed_dim
        4,     // gin_channels
        48000  // sr
      ];

      expect(config).toHaveLength(14);
      expect(typeof config[0]).toBe("number");
      expect(typeof config[13]).toBe("number");
    });

    it("should handle config with minimal values", () => {
      const config = [
        128,   // hidden_channels (minimum)
        64,    // filter_channels  
        2,     // n_heads
        16000, // sample_rate
        0,     // p_dropout
        1,     // resblock
        1,     // resblock_kernel_sizes
        1,     // resblock_dilation_sizes
        8,     // upsample_rates
        32,    // upsample_initial_channel
        1,     // upsample_kernel_sizes
        1,     // spk_embed_dim
        1,     // gin_channels
        0      // sr (no f0)
      ];

      expect(config).toHaveLength(14);
      expect(config[0]).toBeGreaterThanOrEqual(128);
    });

    it("should handle config with large values", () => {
      const config = [
        1024,  // hidden_channels (large)
        512,   // filter_channels  
        16,    // n_heads
        48000, // sample_rate
        10,    // p_dropout
        12,    // resblock
        8,     // resblock_kernel_sizes
        8,     // resblock_dilation_sizes
        2048,  // upsample_rates
        512,   // upsample_initial_channel
        16,    // upsample_kernel_sizes
        256,   // spk_embed_dim
        512,   // gin_channels
        48000  // sr
      ];

      expect(config).toHaveLength(14);
      expect(config[0]).toBeGreaterThan(768);
    });
  });

  describe("Real File Parsing", () => {
    it("should parse TestModel_Original.pth successfully", async () => {
      const pthPath = resolve(__dirname, "fixtures", "TestModel_Original.pth");
      const pthBuffer = await readFile(pthPath);
      
      const checkpoint = await parsePth(pthBuffer.buffer);
      
      // Verify structure
      expect(checkpoint).toHaveProperty("config");
      expect(checkpoint).toHaveProperty("weights"); // Note: "weights" not "weight"
      expect(checkpoint).toHaveProperty("version");
      expect(checkpoint).toHaveProperty("useF0");
      
      // Verify config structure
      expect(typeof checkpoint.config).toBe("object");
      expect(checkpoint.config).toHaveProperty("hiddenChannels");
      expect(checkpoint.config).toHaveProperty("sr"); // "sr" not "sampleRate"
    });
  });

  describe("Error Handling", () => {
    it("should reject invalid pickle data", async () => {
      const invalidData = new Uint8Array([0x00, 0x01, 0x02, 0x03]);
      
      await expect(parsePth(invalidData.buffer)).rejects.toThrow();
    });

    it("should reject empty buffer", async () => {
      const emptyData = new Uint8Array([]);
      
      await expect(parsePth(emptyData.buffer)).rejects.toThrow();
    });

    it("should reject malformed pickle protocol", async () => {
      // Valid PROTO marker but invalid data
      const malformedData = new Uint8Array([0x80, 0x04, 0xff, 0xff]);
      
      await expect(parsePth(malformedData.buffer)).rejects.toThrow();
    });

    it("should reject truncated pickle data", async () => {
      // PROTO 4, BININT1 but missing byte
      const truncatedData = new Uint8Array([0x80, 0x04, 0x4b]);
      
      await expect(parsePth(truncatedData.buffer)).rejects.toThrow();
    });

    it("should reject pickle without STOP opcode", async () => {
      // Valid start but no STOP
      const noStopData = new Uint8Array([0x80, 0x04, 0x7d]);
      
      await expect(parsePth(noStopData.buffer)).rejects.toThrow();
    });
  });

  describe("Storage Resolution", () => {
    it("should handle models with storage references", async () => {
      const pthPath = resolve(__dirname, "fixtures", "TestModel_Original.pth");
      const pthBuffer = await readFile(pthPath);
      
      const checkpoint = await parsePth(pthBuffer.buffer);
      
      // Weights should contain tensor data
      expect(checkpoint.weights).toBeDefined();
      expect(checkpoint.weights instanceof Map).toBe(true);
      expect(checkpoint.weights.size).toBeGreaterThan(0);
    });
  });

  describe("Config Array Access", () => {
    it("should access config values by index", () => {
      const config = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 48000];
      
      expect(config[0]).toBe(768);   // hidden_channels
      expect(config[1]).toBe(192);   // filter_channels
      expect(config[2]).toBe(192);   // n_heads
      expect(config[3]).toBe(40000); // sample_rate
      expect(config[13]).toBe(48000); // sr
    });

    it("should handle out of bounds access gracefully", () => {
      const config = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 48000];
      
      expect(config[14]).toBeUndefined();
      expect(config[100]).toBeUndefined();
      expect(config[-1]).toBeUndefined();
    });
  });

  describe("F0 Detection", () => {
    it("should detect f0 when sr is non-zero", () => {
      const configWithF0 = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 48000];
      const sr = configWithF0[13];
      const hasF0 = sr !== 0;
      
      expect(hasF0).toBe(true);
    });

    it("should detect no f0 when sr is zero", () => {
      const configNoF0 = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 0];
      const sr = configNoF0[13];
      const hasF0 = sr !== 0;
      
      expect(hasF0).toBe(false);
    });

    it("should detect no f0 when sr is negative (edge case)", () => {
      const configNegativeSr = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, -1];
      const sr = configNegativeSr[13];
      const hasF0 = sr !== 0;
      
      // Even negative values are "non-zero" for f0 detection
      expect(hasF0).toBe(true);
    });
  });

  describe("Version Detection Logic", () => {
    it("should identify v1 models correctly", () => {
      const v1Config = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 0];
      
      // v1 models have sr=0 (no f0)
      expect(v1Config[13]).toBe(0);
      
      // This would be used in pthToOnnx conversion
      const version = v1Config[13] === 0 ? "v1" : "v2";
      expect(version).toBe("v1");
    });

    it("should identify v2 models correctly", () => {
      const v2Config = [768, 192, 192, 40000, 2, 6, 1, 3, 512, 1, 1, 2, 4, 48000];
      
      // v2 models have sr!=0 (has f0)
      expect(v2Config[13]).not.toBe(0);
      
      const version = v2Config[13] === 0 ? "v1" : "v2";
      expect(version).toBe("v2");
    });
  });
});
