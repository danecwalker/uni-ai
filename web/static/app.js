const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const messagesEl = document.getElementById('messages');
const form = document.getElementById('chat-form');
const input = document.getElementById('text-input');
const micBtn = document.getElementById('mic-btn');
const modePill = document.getElementById('mode-pill');
const emotionPill = document.getElementById('emotion-pill');
const camToggle = document.getElementById('cam-toggle');
const ttsToggle = document.getElementById('tts-toggle');
const voiceSelect = document.getElementById('voice-select');
const blob = document.getElementById('blob');
const blobLabel = document.getElementById('blob-label');
const blobCaption = document.getElementById('blob-caption');
const chatToggleBtn = document.getElementById('chat-toggle-btn');
const chatDrawer = document.getElementById('chat-drawer');

const STATE_LABELS = {
  idle: 'Idle',
  armed: 'Listening',
  listening: 'Hearing you',
  thinking: 'Thinking',
  talking: 'Speaking',
  triggered: 'Reaching out',
};
function setBlobState(state) {
  if (!STATE_LABELS[state]) return;
  blob.dataset.state = state;
  blobLabel.textContent = STATE_LABELS[state];
  if (window.BlobScene) window.BlobScene.setState(state);
}
function setCaption(text) {
  blobCaption.textContent = text || '';
}

chatToggleBtn.addEventListener('click', () => {
  const show = chatDrawer.hasAttribute('hidden');
  if (show) chatDrawer.removeAttribute('hidden');
  else chatDrawer.setAttribute('hidden', '');
  chatToggleBtn.textContent = show ? 'Hide chat' : 'Show chat';
});
const resetBtn = document.getElementById('reset-btn');
const convoBtn = document.getElementById('convo-btn');
const listeningEl = document.getElementById('listening-indicator');

let lastEmotion = null;
let mediaStream = null;
let emotionTimer = null;
let serverState = { emotion_available: false };

// Hands-free conversation state
let convoActive = false;
let vadController = null;        // MicVadController instance
let busyWithTurn = false;        // pause VAD while transcribing/replying

// Push-to-talk recorder (separate)
let ptt = { recorder: null, chunks: [] };

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setMode(mode) {
  modePill.textContent = mode;
  modePill.className = `pill ${mode}`;
}

function setListening(on, label, active = false) {
  listeningEl.hidden = !on;
  listeningEl.classList.toggle('active', !!(on && active));
  input.style.visibility = on ? 'hidden' : 'visible';
  if (on && label) listeningEl.querySelector('.listening-label').textContent = label;
}

let ttsAudio = null;       // currently-playing <audio>
let ttsAvailable = false;  // backend Groq TTS available

async function speak(text) {
  if (!ttsToggle.checked || !text) return;
  if (ttsAvailable) {
    try { await speakViaGroq(text); return; }
    catch (e) { console.warn('Groq TTS failed, falling back to browser:', e); }
  }
  await speakViaBrowser(text);
}

function stopSpeaking() {
  if (ttsAudio) { try { ttsAudio.pause(); } catch {} ttsAudio = null; }
  try { speechSynthesis.cancel(); } catch {}
}

let ttsAnalyserCtx = null;
let ttsAmpRAF = null;

async function speakViaGroq(text) {
  stopSpeaking();
  const voice = voiceSelect.value || undefined;
  const r = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, voice }),
  });
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).detail || ''; } catch {}
    throw new Error(`tts ${r.status} ${detail}`);
  }
  const blobData = await r.blob();
  if (blobData.size < 100) throw new Error(`tts empty response (${blobData.size}b)`);
  const url = URL.createObjectURL(blobData);
  ttsAudio = new Audio(url);
  ttsAudio.crossOrigin = 'anonymous';
  await new Promise((resolve, reject) => {
    ttsAudio.onended = () => { URL.revokeObjectURL(url); stopAmpLoop(); resolve(); };
    ttsAudio.onerror = () => { URL.revokeObjectURL(url); stopAmpLoop(); reject(new Error('audio element error')); };
    ttsAudio.play().then(() => startAmpLoop(ttsAudio)).catch((err) => { URL.revokeObjectURL(url); reject(err); });
  });
  ttsAudio = null;
}

function startAmpLoop(audioEl) {
  try {
    if (!ttsAnalyserCtx) ttsAnalyserCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (ttsAnalyserCtx.state === 'suspended') ttsAnalyserCtx.resume();
    const src = ttsAnalyserCtx.createMediaElementSource(audioEl);
    const analyser = ttsAnalyserCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    src.connect(ttsAnalyserCtx.destination);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const amp = Math.min(1, rms * 3);
      if (window.BlobScene) window.BlobScene.setAmp(amp);
      ttsAmpRAF = requestAnimationFrame(tick);
    };
    tick();
  } catch (e) {
    // createMediaElementSource only works once per element across contexts; ignore.
  }
}

function stopAmpLoop() {
  if (ttsAmpRAF) { cancelAnimationFrame(ttsAmpRAF); ttsAmpRAF = null; }
  if (window.BlobScene) window.BlobScene.setAmp(0);
}

function speakViaBrowser(text) {
  return new Promise((resolve) => {
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.onend = () => resolve();
      u.onerror = () => resolve();
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    } catch { resolve(); }
  });
}

async function sendChat(text) {
  addMessage('user', text);
  setCaption(text);
  const body = { text };
  if (lastEmotion && lastEmotion.label !== 'neutral') {
    body.emotion_hint = lastEmotion.label;
    body.confidence = lastEmotion.confidence;
  }
  setBlobState('thinking');
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (data.reply) {
    addMessage('assistant', data.reply);
    setCaption(data.reply);
    setBlobState('talking');
    await speak(data.reply);
  }
  setMode(data.mode);
  return data;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  await sendChat(text);
});

resetBtn.addEventListener('click', async () => {
  await stopConversation();
  await fetch('/api/reset', { method: 'POST' });
  messagesEl.innerHTML = '';
  lastEmotion = null;
  setMode('idle');
  addMessage('system', 'Conversation reset.');
});

// --- Webcam + emotion polling ---
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    video.srcObject = mediaStream;
    if (serverState.emotion_available) startEmotionLoop();
  } catch (e) {
    addMessage('system', `Camera/mic unavailable: ${e.message}`);
  }
}

function stopCamera() {
  if (emotionTimer) { clearInterval(emotionTimer); emotionTimer = null; }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
    video.srcObject = null;
  }
}

camToggle.addEventListener('change', () => {
  if (camToggle.checked) startCamera(); else stopCamera();
});

function startEmotionLoop() {
  if (emotionTimer) clearInterval(emotionTimer);
  emotionTimer = setInterval(captureAndSendFrame, 800);
  if (!window._visionTimer) {
    window._visionTimer = setInterval(captureAndSendVisionFrame, 3500);
  }
}

async function captureAndSendVisionFrame() {
  if (!video.videoWidth) return;
  canvas.width = 512;
  canvas.height = Math.round(512 * video.videoHeight / video.videoWidth);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.8));
  if (!blob) return;
  const fd = new FormData();
  fd.append('frame', blob, 'frame.jpg');
  try {
    const r = await fetch('/api/vision/snapshot', { method: 'POST', body: fd });
    const data = await r.json();
    drawBboxOverlay(data, canvas.width, canvas.height);
    if (data && data.person_trigger && data.person_trigger.opener && !convoActive) {
      setMode('conversation');
      setBlobState('triggered');
      addMessage('assistant', data.person_trigger.opener);
      setCaption(data.person_trigger.opener);
      setBlobState('talking');
      await speak(data.person_trigger.opener);
      startConversation(true);
    }
  } catch { /* ignore */ }
}

function drawBboxOverlay(data, sourceW, sourceH) {
  const overlay = document.getElementById('bbox-overlay');
  if (!overlay || !video.videoWidth) return;
  const dw = video.clientWidth;
  const dh = video.clientHeight;
  if (overlay.width !== dw) overlay.width = dw;
  if (overlay.height !== dh) overlay.height = dh;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, dw, dh);

  // Compute the object-fit: cover transform from source-frame coords → display coords.
  const sourceAR = sourceW / sourceH;
  const displayAR = dw / dh;
  let scale, offsetX, offsetY;
  if (displayAR > sourceAR) {
    scale = dw / sourceW;
    offsetX = 0;
    offsetY = (dh - sourceH * scale) / 2;
  } else {
    scale = dh / sourceH;
    offsetX = (dw - sourceW * scale) / 2;
    offsetY = 0;
  }
  // Video element is mirrored via CSS — flip x to match what the user sees.
  const mirror = (x) => dw - x;

  ctx.lineWidth = 1.5;
  ctx.font = '600 11px system-ui, sans-serif';
  ctx.textBaseline = 'bottom';

  const drawBox = (box, colour, label) => {
    const [x1, y1, x2, y2] = box;
    const left   = x1 * scale + offsetX;
    const right  = x2 * scale + offsetX;
    const top    = y1 * scale + offsetY;
    const bottom = y2 * scale + offsetY;
    const xMirrored = mirror(right);
    const w = right - left;
    const h = bottom - top;
    ctx.strokeStyle = colour;
    ctx.strokeRect(xMirrored, top, w, h);
    if (label) drawLabel(ctx, label, xMirrored, top, colour);
  };

  if (data && data.faces) {
    for (const f of data.faces) {
      const known = f.match && f.match.similarity >= 0.85;
      const colour = known ? 'rgba(108, 209, 108, 0.95)' : 'rgba(245, 110, 110, 0.95)';
      let label = known ? f.match.name : 'unknown';
      if (f.match && !known) label = `?${f.match.name} (${f.match.similarity.toFixed(2)})`;
      drawBox(f.bbox, colour, label);
    }
  }
}

function drawLabel(ctx, text, x, y, fill) {
  const padding = 4;
  const metrics = ctx.measureText(text);
  const tw = metrics.width + padding * 2;
  const th = 14;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.fillRect(x, y - th, tw, th);
  ctx.fillStyle = fill;
  ctx.fillText(text, x + padding, y - 2);
}

async function captureAndSendFrame() {
  if (!video.videoWidth) return;
  canvas.width = 320;
  canvas.height = Math.round(320 * video.videoHeight / video.videoWidth);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.7));
  if (!blob) return;
  const fd = new FormData();
  fd.append('frame', blob, 'frame.jpg');
  try {
    const r = await fetch('/api/emotion', { method: 'POST', body: fd });
    const data = await r.json();
    if (!data.available) {
      emotionPill.textContent = 'emotion: model off';
      clearInterval(emotionTimer);
      emotionTimer = null;
      return;
    }
    if (data.observation) {
      lastEmotion = data.observation;
      emotionPill.textContent = `emotion: ${data.observation.label} ${(data.observation.confidence*100|0)}%`;
    } else {
      emotionPill.textContent = 'emotion: no face';
    }
    if (data.trigger && data.trigger.opener && !convoActive) {
      setMode('conversation');
      setBlobState('triggered');
      addMessage('assistant', data.trigger.opener);
      setCaption(data.trigger.opener);
      setBlobState('talking');
      await speak(data.trigger.opener);
      startConversation(true);
    }
  } catch { /* ignore */ }
}

// --- Push-to-talk mic ---
micBtn.addEventListener('mousedown', pttStart);
micBtn.addEventListener('mouseup', pttStop);
micBtn.addEventListener('mouseleave', pttStop);
micBtn.addEventListener('touchstart', (e) => { e.preventDefault(); pttStart(); });
micBtn.addEventListener('touchend', (e) => { e.preventDefault(); pttStop(); });

function pickMime() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
    'audio/mpeg',
    '',
  ];
  for (const m of candidates) {
    if (m === '' || (window.MediaRecorder && MediaRecorder.isTypeSupported(m))) {
      return m;
    }
  }
  return '';
}

async function pttStart() {
  if (convoActive) return;
  if (ptt.recorder && ptt.recorder.state === 'recording') return;
  if (!mediaStream) { addMessage('system', 'Enable the webcam first to use the mic.'); return; }
  // Use audio-only stream so MediaRecorder doesn't try to encode video.
  const audioOnly = new MediaStream(mediaStream.getAudioTracks());
  const mime = pickMime();
  ptt.chunks = [];
  try {
    ptt.recorder = mime ? new MediaRecorder(audioOnly, { mimeType: mime }) : new MediaRecorder(audioOnly);
  } catch (e) {
    addMessage('system', `Recorder unsupported: ${e.message}`);
    return;
  }
  const usedMime = ptt.recorder.mimeType || mime || 'audio/webm';
  const ext = usedMime.includes('mp4') ? 'mp4' : usedMime.includes('mpeg') ? 'mp3' : usedMime.includes('ogg') ? 'ogg' : 'webm';
  ptt.recorder.ondataavailable = (e) => { if (e.data.size) ptt.chunks.push(e.data); };
  ptt.recorder.onstop = async () => {
    const blob = new Blob(ptt.chunks, { type: usedMime });
    if (blob.size < 2000) return;
    await transcribeAndSend(blob, `speech.${ext}`);
  };
  ptt.recorder.start();
  micBtn.classList.add('recording');
}

function pttStop() {
  if (ptt.recorder && ptt.recorder.state === 'recording') {
    ptt.recorder.stop();
    micBtn.classList.remove('recording');
  }
}

// --- Shared: blob → STT → chat ---
async function transcribeAndSend(blob, filename) {
  const fd = new FormData();
  fd.append('audio', blob, filename);
  let data = null;
  try {
    const r = await fetch('/api/stt', { method: 'POST', body: fd });
    data = await r.json();
  } catch (e) {
    addMessage('system', `Transcription failed: ${e.message}`);
  }
  if (data && data.text) {
    const result = await sendChat(data.text);
    if (result && result.mode !== 'conversation') {
      await stopConversation();
    }
  }
}

// --- Hands-free conversation (Silero VAD via @ricky0123/vad-web) ---
convoBtn.addEventListener('click', async () => {
  if (convoActive) await stopConversation();
  else await startConversation(false);
});

async function startConversation(triggered) {
  if (convoActive) return;
  convoActive = true;
  convoBtn.textContent = 'Stop conversation';
  convoBtn.classList.add('active');
  setMode('conversation');
  if (!triggered) addMessage('system', 'Conversation started — speak any time.');
  try {
    if (!vadController) {
      vadController = new window.MicVadController({
        onSpeechStart: () => {
          if (!convoActive || busyWithTurn) return;
          setListening(true, 'Listening…', true);
          setBlobState('listening');
        },
        onSpeechEnd: async (audio) => {
          if (!convoActive || busyWithTurn) return;
          busyWithTurn = true;
          setListening(false);
          setBlobState('thinking');
          vadController.pause();
          try {
            const wav = encodeWav(audio, 16000);
            await transcribeAndSend(wav, 'speech.wav');
          } finally {
            busyWithTurn = false;
            if (convoActive) {
              setTimeout(() => {
                if (convoActive && vadController) {
                  vadController.start().catch(() => {});
                  setListening(true, 'Listening…', false);
                  setBlobState('armed');
                }
              }, 150);
            }
          }
        },
        onVADMisfire: () => { setListening(true, 'Listening…', false); setBlobState('armed'); },
      });
    }
    await vadController.start();
    setListening(true, 'Listening…', false);
    setBlobState('armed');
  } catch (e) {
    addMessage('system', `VAD failed: ${e.message}`);
    console.error(e);
    await stopConversation();
  }
}

async function stopConversation() {
  if (!convoActive) return;
  convoActive = false;
  convoBtn.textContent = 'Start conversation';
  convoBtn.classList.remove('active');
  setListening(false);
  setMode('idle');
  setBlobState('idle');
  setCaption('');
  stopSpeaking();      // kill any in-flight TTS playback
  stopAmpLoop();
  // Tell the backend to drop conversation state too, so it actually goes idle.
  fetch('/api/reset', { method: 'POST' }).catch(() => {});
  if (vadController) vadController.pause();
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function loadVoices() {
  try {
    const r = await fetch('/api/voices');
    const data = await r.json();
    voiceSelect.innerHTML = '';
    if (!data.available || !data.voices || !data.voices.length) {
      const opt = document.createElement('option');
      opt.value = ''; opt.textContent = 'Browser default';
      voiceSelect.appendChild(opt);
      ttsAvailable = false;
      return;
    }
    for (const v of data.voices) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v.replace(/-PlayAI$/, '');
      if (v === data.default) opt.selected = true;
      voiceSelect.appendChild(opt);
    }
    ttsAvailable = true;
  } catch {
    ttsAvailable = false;
  }
}

(async function init() {
  try {
    const r = await fetch('/api/state');
    serverState = await r.json();
    setMode(serverState.mode);
    if (!serverState.llm_available) addMessage('system', 'GROQ_API_KEY not set: chat will not work.');
    if (!serverState.emotion_available) emotionPill.textContent = 'emotion: model off';
  } catch {}
  await loadVoices();
  // blob.js self-mounts; just push the initial state once it's ready.
  const pushIdle = () => { if (window.BlobScene) setBlobState('idle'); };
  if (window.BlobScene) pushIdle(); else setTimeout(pushIdle, 200);
  if (camToggle.checked) startCamera();
})();
