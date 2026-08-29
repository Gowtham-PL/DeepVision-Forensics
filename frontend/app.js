/**
 * DeepVision-Forensics — Frontend Application Controller
 * Handles image ingestion, client-side validation, API communication,
 * and forensic explainability rendering.
 */

// Configurable API Base URL
const API_BASE = window.DEEPVISION_API_BASE || '/api/v1';

// State Management
let selectedFile = null;
let currentAnalysis = null;

// DOM Elements
const systemStatus = document.getElementById('systemStatus');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const deviceBadge = document.getElementById('deviceBadge');

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const dropzoneEmpty = document.getElementById('dropzoneEmpty');
const browseBtn = document.getElementById('browseBtn');
const loadSampleBtn = document.getElementById('loadSampleBtn');

const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');
const previewFilename = document.getElementById('previewFilename');
const previewFilesize = document.getElementById('previewFilesize');
const removeFileBtn = document.getElementById('removeFileBtn');

const fftToggle = document.getElementById('fftToggle');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = analyzeBtn.querySelector('.btn-text');
const btnSpinner = analyzeBtn.querySelector('.btn-spinner');

const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');

const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsContainer = document.getElementById('resultsContainer');

const classificationBadge = document.getElementById('classificationBadge');
const riskBadge = document.getElementById('riskBadge');
const probabilityText = document.getElementById('probabilityText');
const probabilityFill = document.getElementById('probabilityFill');
const authenticityAssessment = document.getElementById('authenticityAssessment');
const spatialSummary = document.getElementById('spatialSummary');

const tabGradCam = document.getElementById('tabGradCam');
const tabFft = document.getElementById('tabFft');
const panelGradCam = document.getElementById('panelGradCam');
const panelFft = document.getElementById('panelFft');

const vizOriginalImage = document.getElementById('vizOriginalImage');
const vizGradCamImage = document.getElementById('vizGradCamImage');
const vizFftImage = document.getElementById('vizFftImage');
const disclaimerText = document.getElementById('disclaimerText');

/**
 * Initialize application
 */
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setupEventListeners();

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('sample') === 'true' || urlParams.get('demo') === 'true' || urlParams.get('mock_upload') === 'true') {
    loadSampleImage();
  }
});

/**
 * Check backend system health and model availability
 */
async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (!response.ok) throw new Error(`Health check returned ${response.status}`);
    
    const data = await response.json();
    if (data.status === 'healthy' && data.model_loaded) {
      statusDot.className = 'status-indicator-dot online';
      statusText.textContent = 'Model Online';
      if (data.device) {
        deviceBadge.textContent = data.device.toUpperCase();
        deviceBadge.style.display = 'inline-block';
      }
    } else {
      statusDot.className = 'status-indicator-dot offline';
      statusText.textContent = 'Model Degraded';
    }
  } catch (err) {
    statusDot.className = 'status-indicator-dot offline';
    statusText.textContent = 'Backend Offline';
    deviceBadge.style.display = 'none';
  }
}

/**
 * Register UI interaction events
 */
function setupEventListeners() {
  // Browse click
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  loadSampleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    loadSampleImage();
  });

  dropzone.addEventListener('click', () => {
    if (!selectedFile) fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop handlers
  ['dragenter', 'dragover'].forEach((event) => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((event) => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFileSelected(dt.files[0]);
    }
  });

  // Remove file
  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  // Analyze button
  analyzeBtn.addEventListener('click', performAnalysis);

  // Visualization Tabs
  tabGradCam.addEventListener('click', () => switchVizTab('gradcam'));
  tabFft.addEventListener('click', () => switchVizTab('fft'));
}

/**
 * Helper: Load synthetic test sample image for instant testing
 */
function loadSampleImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  // Draw colorful test gradient with geometric shapes
  const grad = ctx.createLinearGradient(0, 0, 256, 256);
  grad.addColorStop(0, '#0284c7');
  grad.addColorStop(0.5, '#6366f1');
  grad.addColorStop(1, '#ec4899');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);

  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(128, 128, 55, 0, Math.PI * 2);
  ctx.fill();

  canvas.toBlob((blob) => {
    const file = new File([blob], 'synthetic_test_sample.png', { type: 'image/png' });
    handleFileSelected(file);
  }, 'image/png');
}

/**
 * Handle image file selection
 */
function handleFileSelected(file) {
  hideError();

  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const allowedExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();

  if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(ext)) {
    showError('Unsupported file format. Please upload a valid JPEG, PNG, or WEBP image.');
    return;
  }

  // 10 MB limit
  const maxBytes = 10 * 1024 * 1024;
  if (file.size > maxBytes) {
    showError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds maximum allowed limit of 10 MB.`);
    return;
  }

  selectedFile = file;

  // Render client-side preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    dropzoneEmpty.style.display = 'none';
    previewBox.style.display = 'flex';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

/**
 * Reset file selection state
 */
function resetFileSelection() {
  selectedFile = null;
  fileInput.value = '';
  previewImage.src = '';
  dropzoneEmpty.style.display = 'block';
  previewBox.style.display = 'none';
  analyzeBtn.disabled = true;
  hideError();
}

/**
 * Switch visualization tab
 */
function switchVizTab(tabName) {
  if (tabName === 'gradcam') {
    tabGradCam.classList.add('active');
    tabFft.classList.remove('active');
    panelGradCam.style.display = 'block';
    panelFft.style.display = 'none';
  } else {
    tabFft.classList.add('active');
    tabGradCam.classList.remove('active');
    panelFft.style.display = 'block';
    panelGradCam.style.display = 'none';
  }
}

/**
 * Execute forensic analysis request to backend
 */
async function performAnalysis() {
  if (!selectedFile) return;

  hideError();
  setLoading(true);

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('include_fft', fftToggle.checked ? 'true' : 'false');

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = `Analysis failed with status ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) errorDetail = errJson.detail;
      } catch (_) {}
      throw new Error(errorDetail);
    }

    const report = await response.json();
    renderAnalysisResults(report);

  } catch (err) {
    showError(err.message || 'Unable to complete forensic analysis. Please verify backend connection.');
  } finally {
    setLoading(false);
  }
}

/**
 * Render analysis results to UI
 */
function renderAnalysisResults(report) {
  currentAnalysis = report;

  const pred = report.prediction;
  const isAi = pred.classification_label === 'AI-generated';
  const probPct = (pred.ai_probability * 100).toFixed(1) + '%';

  // 1. Classification & Badges
  classificationBadge.textContent = pred.classification_label;
  classificationBadge.className = `classification-badge ${isAi ? 'ai' : 'real'}`;

  riskBadge.textContent = `${pred.risk_indicator} RISK`;
  riskBadge.className = `risk-pill ${pred.risk_indicator.toLowerCase()}`;

  // 2. Probability Gauge
  probabilityText.textContent = probPct;
  probabilityFill.style.width = `${Math.min(Math.max(pred.ai_probability * 100, 2), 100)}%`;

  // 3. Assessment & Explanations
  authenticityAssessment.textContent = pred.authenticity_assessment;
  spatialSummary.textContent = report.evidence.spatial_summary;

  // 4. Visualizations
  vizOriginalImage.src = previewImage.src;

  if (report.visualizations.gradcam_heatmap) {
    vizGradCamImage.src = report.visualizations.gradcam_heatmap;
  }

  if (report.visualizations.fft_spectrum) {
    vizFftImage.src = report.visualizations.fft_spectrum;
    tabFft.style.display = 'inline-block';
  } else {
    tabFft.style.display = 'none';
    switchVizTab('gradcam');
  }

  // 5. Disclaimer
  if (report.disclaimer) {
    disclaimerText.textContent = report.disclaimer;
  }

  // Switch display from placeholder to results
  resultsPlaceholder.style.display = 'none';
  resultsContainer.style.display = 'flex';
}

/**
 * Set loading state on action button
 */
function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  btnText.textContent = isLoading ? 'Analyzing Image...' : 'Analyze Image';
  btnSpinner.style.display = isLoading ? 'inline-block' : 'none';
}

/**
 * Display error banner
 */
function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.style.display = 'flex';
}

/**
 * Hide error banner
 */
function hideError() {
  errorBanner.style.display = 'none';
}

/**
 * Helper: Format byte counts
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
