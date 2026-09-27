const API_URL = "http://127.0.0.1:8000";

let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let pcmBuffers = [];
let recordingLength = 0;
let isRecording = false;

let recordedBlob = null;
let uploadedFile = null;
let timerInterval = null;
let secondsElapsed = 0;

const tabRecord = document.getElementById("tab-record");
const tabUpload = document.getElementById("tab-upload");
const panelRecord = document.getElementById("panel-record");
const panelUpload = document.getElementById("panel-upload");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnAnalyze = document.getElementById("btn-analyze");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileSelectedName = document.getElementById("file-selected-name");
const canvas = document.getElementById("visualizer");
const canvasCtx = canvas.getContext("2d");
const timerTag = document.getElementById("recording-timer");
const btnRefresh = document.getElementById("btn-refresh");

// Tab Switching
tabRecord.addEventListener("click", () => {
  tabRecord.classList.add("active");
  tabUpload.classList.remove("active");
  panelRecord.classList.add("active");
  panelUpload.classList.remove("active");
  uploadedFile = null;
  fileSelectedName.textContent = "";
  btnAnalyze.disabled = !recordedBlob;
});

tabUpload.addEventListener("click", () => {
  tabUpload.classList.add("active");
  tabRecord.classList.remove("active");
  panelUpload.classList.add("active");
  panelRecord.classList.remove("active");
  btnAnalyze.disabled = !uploadedFile;
});

// File Upload Handler
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--primary)";
});
dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "var(--border-color)";
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  uploadedFile = file;
  fileSelectedName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  btnAnalyze.disabled = false;
}

// Convert PCM Samples to Real WAV File in Memory
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit PCM samples with clipping
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// Visualizer Setup
function drawWave(inputData) {
  canvasCtx.fillStyle = "#161c28";
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = "#6366f1";
  canvasCtx.beginPath();

  const sliceWidth = canvas.width / inputData.length;
  let x = 0;
  for (let i = 0; i < inputData.length; i++) {
    const v = (inputData[i] + 1) / 2;
    const y = v * canvas.height;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }
  canvasCtx.stroke();
}

function clearCanvas() {
  canvasCtx.fillStyle = "#161c28";
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
}
clearCanvas();

// Microphone Recording via Raw Web Audio API
btnStart.addEventListener("click", async () => {
  pcmBuffers = [];
  recordingLength = 0;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      pcmBuffers.push(new Float32Array(input));
      recordingLength += input.length;
      drawWave(input);
    };

    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    btnStart.disabled = true;
    btnStop.disabled = false;

    secondsElapsed = 0;
    timerInterval = setInterval(() => {
      secondsElapsed++;
      const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
      const secs = String(secondsElapsed % 60).padStart(2, "0");
      timerTag.textContent = `${mins}:${secs}`;
    }, 1000);
  } catch (err) {
    alert("Microphone Error: " + err.message);
  }
});

btnStop.addEventListener("click", () => {
  if (!isRecording) return;
  isRecording = false;
  clearInterval(timerInterval);

  btnStart.disabled = false;
  btnStop.disabled = true;

  if (scriptProcessor) scriptProcessor.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());

  // Flatten PCM buffers
  const fullSamples = new Float32Array(recordingLength);
  let offset = 0;
  for (let i = 0; i < pcmBuffers.length; i++) {
    fullSamples.set(pcmBuffers[i], offset);
    offset += pcmBuffers[i].length;
  }

  // Encode clean standard WAV file
  recordedBlob = encodeWAV(fullSamples, 16000);
  btnAnalyze.disabled = false;
  clearCanvas();
});

// Run Security Scan
btnAnalyze.addEventListener("click", async () => {
  btnAnalyze.disabled = true;
  btnAnalyze.textContent = "Scanning Audio Biometrics...";

  const formData = new FormData();
  if (tabRecord.classList.contains("active") && recordedBlob) {
    formData.append("file", recordedBlob, "live_recording.wav");
  } else if (uploadedFile) {
    formData.append("file", uploadedFile, uploadedFile.name);
  }

  try {
    const res = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned status ${res.status}`);
    }
    const data = await res.json();
    renderAnalysis(data);
    loadHistory();
  } catch (err) {
    alert("Scan Failed: " + err.message);
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = "Run Security Scan";
  }
});

function renderAnalysis(data) {
  document.getElementById("placeholder-view").classList.add("hidden");
  document.getElementById("analysis-view").classList.remove("hidden");

  const riskBanner = document.getElementById("risk-banner");
  riskBanner.className = `risk-banner risk-${data.risk_assessment.risk_level.toLowerCase()}`;

  document.getElementById("verdict-text").textContent = data.risk_assessment.verdict;
  document.getElementById("score-val").textContent = `${data.ai_generated_probability}%`;
  document.getElementById("meter-percent").textContent = `${data.ai_generated_probability}%`;

  const meterFill = document.getElementById("meter-fill");
  meterFill.style.width = `${data.ai_generated_probability}%`;
  meterFill.style.backgroundColor = data.risk_assessment.risk_color;

  document.getElementById("recommendation-text").textContent = data.risk_assessment.recommendation;
  document.getElementById("latency-tag").textContent = `${data.latency_ms} ms`;

  const challengeContainer = document.getElementById("challenge-container");
  const challengeList = document.getElementById("challenge-list");
  challengeList.innerHTML = "";

  if (data.risk_assessment.challenge_questions && data.risk_assessment.challenge_questions.length > 0) {
    challengeContainer.classList.remove("hidden");
    data.risk_assessment.challenge_questions.forEach((q) => {
      const li = document.createElement("li");
      li.textContent = q;
      challengeList.appendChild(li);
    });
  } else {
    challengeContainer.classList.add("hidden");
  }
}

async function loadHistory() {
  try {
    const res = await fetch(`${API_URL}/history`);
    if (!res.ok) return;
    const items = await res.json();
    const tbody = document.getElementById("history-rows");
    tbody.innerHTML = "";

    if (items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">No prior audit records.</td></tr>`;
      return;
    }

    items.forEach((row) => {
      const tr = document.createElement("tr");
      const badgeClass = `badge-${row.risk_level.toLowerCase()}`;
      tr.innerHTML = `
        <td>${row.timestamp}</td>
        <td>${row.filename}</td>
        <td>${row.probability}%</td>
        <td><span class="badge-pill ${badgeClass}">${row.risk_level}</span></td>
        <td>${row.latency_ms} ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error("History load error:", e);
  }
}

btnRefresh.addEventListener("click", loadHistory);
loadHistory();