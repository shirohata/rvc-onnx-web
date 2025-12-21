
import { describe, it, expect, vi } from 'vitest';
import { OnnxDataType } from '../src/types';
import { 
  linear, 
  attrFloats, 
  attrString, 
  valueInfo, 
  initializer, 
  node,
  conv1d,
  convTranspose1d,
  linearNodes,
  matmul,
  layerNorm,
  uniqueName,
  resetNameCounter,
  neg,
  sqrt,
  log,
  pow,
  cos,
  sin,
  clip,
  reduceMean,
  pad,
  randomNormalLike,
  split
} from '../src/onnx-builder';

describe('ONNX Builder Helpers', () => {
  it('should create float array attribute', () => {
    const attr = attrFloats('test', [1.0, 2.0]);
    expect(attr).toEqual({
      name: 'test',
      type: 'FLOATS',
      floatsValue: [1.0, 2.0]
    });
  });

  it('should create string attribute', () => {
    const attr = attrString('test', 'value');
    expect(attr).toEqual({
      name: 'test',
      type: 'STRING',
      stringValue: 'value'
    });
  });

  it('should create deprecated linear node', () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const n = linear('in', 'w', 'b', 'out');
    expect(n.opType).toBe('Gemm');
    expect(n.inputs).toEqual(['in', 'w', 'b']);
    expect(n.outputs).toEqual(['out']);
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it('should create linear node without bias', () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const n = linear('in', 'w', null, 'out');
    expect(n.opType).toBe('Gemm');
    expect(n.inputs).toEqual(['in', 'w']);
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it('should create value info with dynamic dimensions', () => {
    const info = valueInfo('test', OnnxDataType.FLOAT, ['batch', 10]);
    expect(info.name).toBe('test');
    expect(info.elemType).toBe(OnnxDataType.FLOAT);
    expect(info.shape).toEqual([
      { dimParam: 'batch' },
      { dimValue: 10n }
    ]);
    // Check BigInt value
    expect(info.shape[1].dimValue).toBe(10n);
  });

  it('should generate unique names', () => {
    resetNameCounter();
    expect(uniqueName('test')).toBe('test_0');
    expect(uniqueName('test')).toBe('test_1');
  });

  it('should create math nodes', () => {
    expect(neg('in', 'out').opType).toBe('Neg');
    expect(sqrt('in', 'out').opType).toBe('Sqrt');
    expect(log('in', 'out').opType).toBe('Log');
    expect(pow('in', 'exp', 'out').opType).toBe('Pow');
    expect(cos('in', 'out').opType).toBe('Cos');
    expect(sin('in', 'out').opType).toBe('Sin');
  });

  it('should create clip node', () => {
    const n = clip('in', 'min', 'max', 'out');
    expect(n.opType).toBe('Clip');
    expect(n.inputs).toEqual(['in', 'min', 'max']);
  });

  it('should create clip node with defaults', () => {
    const n = clip('in', null, null, 'out');
    expect(n.opType).toBe('Clip');
    // Implementation: if (min) inputs.push(min); else inputs.push("");
    // if (max) inputs.push(max); -> max is null, so nothing pushed
    expect(n.inputs).toEqual(['in', '']);
  });

  it('should create reduceMean node', () => {
    const n = reduceMean('in', 'out', [1]);
    expect(n.opType).toBe('ReduceMean');
    expect(n.attributes[0].name).toBe('axes');
    expect(n.attributes[0].intsValue).toEqual([1n]);
  });

  it('should create pad node', () => {
    const n = pad('in', 'pads', 'out');
    expect(n.opType).toBe('Pad');
    expect(n.inputs).toEqual(['in', 'pads']);
  });

  it('should create pad node with constant', () => {
    const n = pad('in', 'pads', 'out', 'constant', 'val');
    expect(n.opType).toBe('Pad');
    expect(n.inputs).toEqual(['in', 'pads', 'val']);
  });

  it('should create randomNormalLike node', () => {
    const n = randomNormalLike('in', 'out');
    expect(n.opType).toBe('RandomNormalLike');
  });

  it('should create split node', () => {
    const n = split('in', ['out1', 'out2']);
    expect(n.opType).toBe('Split');
    expect(n.outputs).toEqual(['out1', 'out2']);
  });

  it('should create split node with sizes', () => {
    const n = split('in', ['out1', 'out2'], 0, 'sizes');
    expect(n.opType).toBe('Split');
    expect(n.inputs).toEqual(['in', 'sizes']);
  });
});
