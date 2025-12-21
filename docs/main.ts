/**
 * RVC to ONNX Converter - Demo App
 */

import { pthToOnnx, parsePth } from 'rvc-onnx-web';

// DOM Elements
const uploadZone = document.getElementById('uploadZone')!;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const fileName = document.getElementById('fileName')!;
const configCard = document.getElementById('configCard')!;
const configGrid = document.getElementById('configGrid')!;
const convertCard = document.getElementById('convertCard')!;
const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement;
const progressContainer = document.getElementById('progressContainer')!;
const progressFill = document.getElementById('progressFill')!;
const progressText = document.getElementById('progressText')!;
const logCard = document.getElementById('logCard')!;
const logContainer = document.getElementById('logContainer')!;
const downloadCard = document.getElementById('downloadCard')!;
const downloadBtn = document.getElementById('downloadBtn')!;
const stats = document.getElementById('stats')!;

// State
let currentFile: File | null = null;
let parsedModel: Awaited<ReturnType<typeof parsePth>> | null = null;
let onnxBlob: Blob | null = null;

// Event Listeners
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer?.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files?.length) {
    handleFile(fileInput.files[0]);
  }
});
convertBtn.addEventListener('click', startConversion);
downloadBtn.addEventListener('click', downloadOnnx);

// Logger
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
}

function setProgress(percent: number, text: string) {
  progressFill.style.width = `${percent}%`;
  progressText.textContent = text;
}

// Handle file upload
async function handleFile(file: File) {
  if (!file.name.endsWith('.pth')) {
    alert('Please upload a .pth file');
    return;
  }
  
  currentFile = file;
  uploadZone.classList.add('has-file');
  fileName.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  
  // Reset state
  logContainer.innerHTML = '';
  downloadCard.classList.add('hidden');
  onnxBlob = null;
  
  // Parse the file
  logCard.classList.remove('hidden');
  log(`Loading ${file.name}...`);
  
  try {
    const arrayBuffer = await file.arrayBuffer();
    log(`File loaded (${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB)`);
    
    log('Parsing PyTorch model...');
    parsedModel = await parsePth(arrayBuffer);
    
    log(`Found ${parsedModel.weights.size} weight tensors`, 'success');
    log(`Model version: ${parsedModel.version}`, 'info');
    log(`Uses F0 (pitch): ${parsedModel.useF0}`, 'info');
    
    // Extract and display config
    displayConfig(parsedModel.config);
    
    configCard.classList.remove('hidden');
    convertCard.classList.remove('hidden');
    
    log('Model parsed successfully! Ready to convert.', 'success');
  } catch (err) {
    log(`Error parsing model: ${(err as Error).message}`, 'error');
    console.error(err);
  }
}

// Display model configuration
function displayConfig(config: Record<string, unknown>) {
  const displayItems = [
    { label: 'Hidden Channels', key: 'hiddenChannels' },
    { label: 'Inter Channels', key: 'interChannels' },
    { label: 'Filter Channels', key: 'filterChannels' },
    { label: 'Encoder Layers', key: 'nLayers' },
    { label: 'Attention Heads', key: 'nHeads' },
    { label: 'Sample Rate', key: 'sr' },
    { label: 'Speaker Embed Dim', key: 'spkEmbedDim' },
    { label: 'Upsample Rates', key: 'upsampleRates', isArray: true },
    { label: 'Upsample Kernels', key: 'upsampleKernelSizes', isArray: true },
    { label: 'ResBlock Kernels', key: 'resblockKernelSizes', isArray: true },
  ];
  
  configGrid.innerHTML = '';
  
  for (const item of displayItems) {
    const value = config[item.key];
    if (value === undefined) continue;
    
    const div = document.createElement('div');
    div.className = 'config-item';
    div.innerHTML = `
      <div class="config-label">${item.label}</div>
      <div class="config-value ${item.isArray ? 'array' : ''}">${
        item.isArray ? JSON.stringify(value) : value
      }</div>
    `;
    configGrid.appendChild(div);
  }
}

// Start conversion
async function startConversion() {
  if (!parsedModel || !currentFile) {
    alert('Please upload a model first');
    return;
  }
  
  convertBtn.disabled = true;
  progressContainer.classList.add('visible');
  
  const startTime = performance.now();
  
  try {
    setProgress(5, 'Building ONNX graph...');
    log('Starting ONNX conversion...');
    
    // Give UI a chance to update
    await new Promise(resolve => setTimeout(resolve, 10));
    
    setProgress(20, 'Building text encoder...');
    log('Building text encoder...');
    await new Promise(resolve => setTimeout(resolve, 10));
    
    setProgress(40, 'Building flow decoder...');
    log('Building flow decoder...');
    await new Promise(resolve => setTimeout(resolve, 10));
    
    setProgress(60, 'Building HiFi-GAN vocoder...');
    log('Building HiFi-GAN vocoder...');
    await new Promise(resolve => setTimeout(resolve, 10));
    
    setProgress(80, 'Serializing ONNX model...');
    log('Serializing to ONNX format...');
    
    // Convert
    const onnxBytes = await pthToOnnx(await currentFile.arrayBuffer(), {
      opsetVersion: 17,
      phoneLen: 100
    });
    
    onnxBlob = new Blob([onnxBytes], { type: 'application/octet-stream' });
    
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    const sizeInMB = (onnxBlob.size / 1024 / 1024).toFixed(2);
    
    setProgress(100, 'Complete!');
    log(`Conversion complete in ${elapsed}s`, 'success');
    log(`Output size: ${sizeInMB} MB`, 'success');
    
    // Show download section
    downloadCard.classList.remove('hidden');
    stats.innerHTML = `
      <span>⏱️ ${elapsed}s</span>
      <span>📦 ${sizeInMB} MB</span>
      <span>⚖️ ${parsedModel.weights.size} weights</span>
    `;
    
  } catch (err) {
    log(`Conversion failed: ${(err as Error).message}`, 'error');
    console.error(err);
    setProgress(0, 'Failed');
  }
  
  convertBtn.disabled = false;
}

// Download ONNX file
function downloadOnnx() {
  if (!onnxBlob || !currentFile) return;
  
  const baseName = currentFile.name.replace('.pth', '');
  const url = URL.createObjectURL(onnxBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${baseName}.onnx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  log(`Downloaded ${baseName}.onnx`, 'success');
}
