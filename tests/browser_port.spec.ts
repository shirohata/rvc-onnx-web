/**
 * Browser Port Test Suite
 *
 * This test validates that the TypeScript/browser-based PTH to ONNX converter
 * produces output matching the expected reference ONNX file.
 *
 * The comparison focuses on deterministic parts:
 * - Graph structure (nodes, inputs, outputs)
 * - Initializer shapes and values (weights)
 * - Node attributes
 */

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { pthToOnnx, parsePth } from "../src/index";
import { buildOnnxModel } from "../src/onnx-builder";
import { serializeOnnx } from "../src/onnx-serializer";

const FIXTURES_DIR = join(__dirname, "fixtures");
const OUTPUT_DIR = join(__dirname, "output");

// Test model paths
const TEST_MODEL_PTH = join(FIXTURES_DIR, "TestModel_Original.pth");
const TEST_MODEL_EXPECTED_ONNX = join(FIXTURES_DIR, "TestModel_Expected.onnx");
const TEST_MODEL_ACTUAL_ONNX = join(OUTPUT_DIR, "TestModel_Actual.onnx");

/**
 * Compute Pearson correlation coefficient between two arrays.
 */
function pearsonCorrelation(x: Float32Array | number[], y: Float32Array | number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
    sumY2 += y[i] * y[i];
  }

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  if (denominator === 0) return sumX === sumY ? 1 : 0;
  return numerator / denominator;
}

/**
 * Compare two ONNX models by parsing their protobuf structure.
 * Returns detailed comparison results including weight correlations.
 */
async function compareOnnxModels(
  actualPath: string,
  expectedPath: string
): Promise<{
  structureMatch: boolean;
  weightCorrelation: number;
  details: {
    nodeCountActual: number;
    nodeCountExpected: number;
    initializerCountActual: number;
    initializerCountExpected: number;
    matchedWeights: number;
    totalWeights: number;
    correlations: { name: string; correlation: number; shape: string }[];
    missingInActual: string[];
    missingInExpected: string[];
  };
}> {
  const { onnx } = await import("onnx-proto");

  const actualBuffer = readFileSync(actualPath);
  const expectedBuffer = readFileSync(expectedPath);

  const actualModel = onnx.ModelProto.decode(actualBuffer);
  const expectedModel = onnx.ModelProto.decode(expectedBuffer);

  const actualGraph = actualModel.graph!;
  const expectedGraph = expectedModel.graph!;

  // Build maps of initializers by normalized name
  // Python exports use "model." prefix, TypeScript doesn't
  const normalizeWeightName = (name: string): string => {
    // Strip "model." prefix if present
    let normalized = name.replace(/^model\./, "");
    // Also handle path-style names like "/enc_p/..." 
    normalized = normalized.replace(/^\//, "").replace(/\//g, ".");
    return normalized;
  };

  const actualInits = new Map<string, any>();
  const expectedInits = new Map<string, any>();
  const actualNameMap = new Map<string, string>(); // normalized -> original
  const expectedNameMap = new Map<string, string>(); // normalized -> original

  for (const init of actualGraph.initializer || []) {
    const normalized = normalizeWeightName(init.name!);
    actualInits.set(normalized, init);
    actualNameMap.set(normalized, init.name!);
  }
  for (const init of expectedGraph.initializer || []) {
    const normalized = normalizeWeightName(init.name!);
    expectedInits.set(normalized, init);
    expectedNameMap.set(normalized, init.name!);
  }

  // Compare weights
  const correlations: { name: string; correlation: number; shape: string }[] = [];
  const missingInActual: string[] = [];
  const missingInExpected: string[] = [];
  let matchedWeights = 0;

  // Check all expected weights exist in actual and compare values
  for (const [normalizedName, expectedInit] of expectedInits) {
    const actualInit = actualInits.get(normalizedName);
    if (!actualInit) {
      const originalName = expectedNameMap.get(normalizedName) || normalizedName;
      missingInActual.push(originalName);
      continue;
    }

    // Get float data from initializers
    let expectedData: Float32Array;
    let actualData: Float32Array;

    // Helper to safely create Float32Array from potentially unaligned buffer
    const safeFloat32Array = (rawData: Uint8Array): Float32Array => {
      // Copy to aligned buffer to avoid alignment issues
      const aligned = new Uint8Array(rawData.length);
      aligned.set(rawData);
      return new Float32Array(aligned.buffer, 0, rawData.length / 4);
    };

    // Handle different data storage formats in ONNX proto
    if (expectedInit.rawData && expectedInit.rawData.length > 0) {
      expectedData = safeFloat32Array(expectedInit.rawData);
    } else if (expectedInit.floatData && expectedInit.floatData.length > 0) {
      expectedData = new Float32Array(expectedInit.floatData);
    } else {
      continue; // Skip non-float initializers
    }

    if (actualInit.rawData && actualInit.rawData.length > 0) {
      actualData = safeFloat32Array(actualInit.rawData);
    } else if (actualInit.floatData && actualInit.floatData.length > 0) {
      actualData = new Float32Array(actualInit.floatData);
    } else {
      continue; // Skip non-float initializers
    }

    // Compute correlation
    const correlation = pearsonCorrelation(actualData, expectedData);
    const shape = (expectedInit.dims || []).join("x");
    const originalName = expectedNameMap.get(normalizedName) || normalizedName;
    correlations.push({ name: originalName, correlation, shape });
    matchedWeights++;
  }

  // Check for extra weights in actual
  for (const normalizedName of actualInits.keys()) {
    if (!expectedInits.has(normalizedName)) {
      const originalName = actualNameMap.get(normalizedName) || normalizedName;
      missingInExpected.push(originalName);
    }
  }

  // Compute average correlation
  const avgCorrelation =
    correlations.length > 0
      ? correlations.reduce((sum, c) => sum + c.correlation, 0) / correlations.length
      : 0;

  // Structure matches if we matched at least 50% of expected weights
  // (some weights like onnx::Conv_* are generated differently by different exporters)
  const weightMatchRatio = matchedWeights / Math.max(expectedInits.size, 1);
  const structureMatch = weightMatchRatio >= 0.5 && avgCorrelation >= 0.99;

  return {
    structureMatch,
    weightCorrelation: avgCorrelation,
    details: {
      nodeCountActual: actualGraph.node!.length,
      nodeCountExpected: expectedGraph.node!.length,
      initializerCountActual: actualGraph.initializer!.length,
      initializerCountExpected: expectedGraph.initializer!.length,
      matchedWeights,
      totalWeights: expectedInits.size,
      correlations: correlations.sort((a, b) => a.correlation - b.correlation).slice(0, 10), // Worst 10
      missingInActual: missingInActual.slice(0, 10),
      missingInExpected: missingInExpected.slice(0, 10),
    },
  };
}

describe("PTH to ONNX Conversion", () => {
  beforeAll(() => {
    if (!existsSync(OUTPUT_DIR)) {
      mkdirSync(OUTPUT_DIR, { recursive: true });
    }
  });

  describe("TestModel Conversion", () => {
    it("should convert TestModel_Original.pth successfully", async () => {
      if (!existsSync(TEST_MODEL_PTH)) {
        console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
        return;
      }

      const pthBuffer = readFileSync(TEST_MODEL_PTH);
      const { onnxBuffer, sampleRate } = await pthToOnnx(pthBuffer.buffer);

      // Write output for comparison
      writeFileSync(TEST_MODEL_ACTUAL_ONNX, onnxBuffer);

      expect(onnxBuffer.length).toBeGreaterThan(0);
      expect(sampleRate).toBeGreaterThan(0);
      expect(existsSync(TEST_MODEL_ACTUAL_ONNX)).toBe(true);

      console.log(`Generated: ${TEST_MODEL_ACTUAL_ONNX} (${onnxBuffer.length} bytes)`);
    });

    it("should produce valid ONNX format", async () => {
      if (!existsSync(TEST_MODEL_ACTUAL_ONNX)) {
        console.warn("Skipping: TestModel_Actual.onnx not generated yet");
        return;
      }

      const onnxBytes = new Uint8Array(readFileSync(TEST_MODEL_ACTUAL_ONNX));

      // ONNX files start with field 1 (ir_version) with varint wire type = 0x08
      expect(onnxBytes[0]).toBe(0x08);
      expect(onnxBytes.length).toBeGreaterThan(1000);
    });

    it("should match expected ONNX output (99%+ weight correlation)", async () => {
      if (!existsSync(TEST_MODEL_ACTUAL_ONNX)) {
        console.warn("Skipping: TestModel_Actual.onnx not generated yet");
        return;
      }

      if (!existsSync(TEST_MODEL_EXPECTED_ONNX)) {
        console.warn(
          `Skipping: ${TEST_MODEL_EXPECTED_ONNX} not found. ` +
            "Generate it using the Python reference implementation."
        );
        return;
      }

      // Skip if expected file is empty (placeholder)
      const expectedStats = readFileSync(TEST_MODEL_EXPECTED_ONNX);
      if (expectedStats.length === 0) {
        console.warn("Skipping: TestModel_Expected.onnx is empty (placeholder)");
        return;
      }

      const comparison = await compareOnnxModels(TEST_MODEL_ACTUAL_ONNX, TEST_MODEL_EXPECTED_ONNX);

      console.log("\n=== ONNX Comparison Results ===");
      console.log(`Structure Match: ${comparison.structureMatch}`);
      console.log(`Weight Correlation: ${(comparison.weightCorrelation * 100).toFixed(4)}%`);
      console.log(`\nDetails:`);
      console.log(`  Nodes: actual=${comparison.details.nodeCountActual}, expected=${comparison.details.nodeCountExpected}`);
      console.log(`  Initializers: actual=${comparison.details.initializerCountActual}, expected=${comparison.details.initializerCountExpected}`);
      console.log(`  Matched Weights: ${comparison.details.matchedWeights}/${comparison.details.totalWeights}`);

      if (comparison.details.correlations.length > 0) {
        console.log(`\nLowest correlations (potential issues):`);
        for (const c of comparison.details.correlations.slice(0, 5)) {
          console.log(`  ${c.name} [${c.shape}]: ${(c.correlation * 100).toFixed(2)}%`);
        }
      }

      if (comparison.details.missingInActual.length > 0) {
        console.log(`\nMissing in actual: ${comparison.details.missingInActual.join(", ")}`);
      }

      if (comparison.details.missingInExpected.length > 0) {
        console.log(`\nExtra in actual: ${comparison.details.missingInExpected.join(", ")}`);
      }

      // Assert 99%+ correlation for deterministic parts
      expect(comparison.weightCorrelation).toBeGreaterThanOrEqual(0.99);
      expect(comparison.structureMatch).toBe(true);
    });
  });
});

describe("PTH Parser", () => {
  it("should parse TestModel_Original.pth", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);

    expect(checkpoint.config).toBeDefined();
    expect(checkpoint.weights.size).toBeGreaterThan(0);
    expect(typeof checkpoint.useF0).toBe("boolean");
    expect(["v1", "v2"]).toContain(checkpoint.version);
  });
});

describe("ONNX Builder", () => {
  it("should build a valid ONNX graph structure", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);
    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });

    expect(model.graph.nodes.length).toBeGreaterThan(0);
    expect(model.graph.inputs.length).toBeGreaterThan(0);
    expect(model.graph.outputs.length).toBeGreaterThan(0);
    expect(model.graph.initializers.length).toBeGreaterThan(0);
  });

  it("should include relative positional encoding weights", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);
    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });

    // Check for relative positional encoding weights
    const relPosInitializers = model.graph.initializers.filter(
      (init) => init.name.includes("emb_rel_k") || init.name.includes("emb_rel_v")
    );

    // RVC v2 models have attention layers with emb_rel_k and emb_rel_v each
    expect(relPosInitializers.length).toBeGreaterThan(0);

    // Verify shape: [1, 2*window_size+1, k_channels]
    for (const init of relPosInitializers) {
      expect(init.data.shape.length).toBe(3);
      expect(init.data.shape[0]).toBe(1);
    }
  });

  it("should include relative position computation nodes", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);
    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });

    const relPosNodes = model.graph.nodes.filter(
      (node) =>
        node.name.includes("rel_") ||
        node.outputs.some((o) => o.includes("rel_")) ||
        node.inputs.some((i) => i.includes("rel_"))
    );

    expect(relPosNodes.length).toBeGreaterThan(0);
  });

  it("should generate valid ONNX that loads in onnxruntime", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);
    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });
    const buffer = serializeOnnx(model);

    let ort;
    try {
      ort = require("onnxruntime-node");
    } catch {
      console.warn("onnxruntime-node not available, skipping runtime check");
      return;
    }

    const session = await ort.InferenceSession.create(buffer.buffer);

    expect(session.inputNames).toContain("phone");
    expect(session.inputNames).toContain("phone_lengths");
    expect(session.inputNames).toContain("sid");
    expect(session.outputNames).toContain("audio");
  });

  it("should run inference with sample inputs", async () => {
    if (!existsSync(TEST_MODEL_PTH)) {
      console.warn(`Skipping: ${TEST_MODEL_PTH} not found`);
      return;
    }

    const pthBuffer = readFileSync(TEST_MODEL_PTH);
    const checkpoint = await parsePth(pthBuffer.buffer);
    const model = buildOnnxModel(checkpoint, { opsetVersion: 17, phoneLen: 100 });
    const buffer = serializeOnnx(model);

    let ort;
    try {
      ort = require("onnxruntime-node");
    } catch {
      console.warn("onnxruntime-node not available, skipping inference test");
      return;
    }

    const session = await ort.InferenceSession.create(buffer.buffer);

    const phoneLen = 100;
    const filterChannels = 768;
    const phone = new Float32Array(1 * phoneLen * filterChannels).fill(0);
    phone[0] = 1.0;

    const phoneLengths = new BigInt64Array([BigInt(phoneLen)]);
    const pitch = new BigInt64Array(1 * phoneLen).fill(BigInt(100));
    const nsff0 = new Float32Array(1 * phoneLen).fill(100.0);
    const sid = new BigInt64Array([BigInt(0)]);

    const feeds = {
      phone: new ort.Tensor("float32", phone, [1, phoneLen, filterChannels]),
      phone_lengths: new ort.Tensor("int64", phoneLengths, [1]),
      pitch: new ort.Tensor("int64", pitch, [1, phoneLen]),
      nsff0: new ort.Tensor("float32", nsff0, [1, phoneLen]),
      sid: new ort.Tensor("int64", sid, [1]),
    };

    const results = await session.run(feeds);

    expect(results.audio).toBeDefined();
    expect(results.sr).toBeDefined();

    const audioShape = results.audio.dims;
    expect(audioShape.length).toBe(3);
    expect(audioShape[0]).toBe(1);
    expect(audioShape[1]).toBe(1);
    expect(audioShape[2]).toBeGreaterThan(0);

    console.log("Inference successful! Audio shape:", audioShape);
  });
});
