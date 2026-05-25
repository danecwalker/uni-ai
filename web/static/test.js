const permPill = document.getElementById('perm-pill');
const ctxPill = document.getElementById('ctx-pill');
const ctxSr = document.getElementById('ctx-sr');
const vadPill = document.getElementById('vad-pill');
const frameCount = document.getElementById('frame-count');
const frameRate = document.getElementById('frame-rate');
const rmsBar = document.getElementById('rms-bar');
const rmsVal = document.getElementById('rms-val');
const vadBar = document.getElementById('vad-bar');
const vadVal = document.getElementById('vad-val');
const vadIndicator = document.getElementById('vad-indicator');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const recordBtn = document.getElementById('record-btn');
const playback = document.getElementById('playback');
const logEl = document.getElementById('log');

let controller = null;
let frames = 0;
let lastRateCheck = 0;
let framesAtLastCheck = 0;
let lastSpeechAudio = null; // Float32Array @ 16kHz

function log(msg) {
  const time = new Date().toTimeString().slice(0, 8);
  logEl.textContent += `[${time}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setPill(el, text, cls) {
  el.textContent = text;
  el.className = `pill ${cls || ''}`;
}

async function start() {
  if (controller) { log('already running'); return; }
  setPill(vadPill, 'loading…', 'warn');
  setPill(ctxPill, 'running', 'ok');
  ctxSr.textContent = '16000 Hz';
  frames = 0;
  framesAtLastCheck = 0;
  lastRateCheck = performance.now();

  controller = new window.MicVadController({
    onSpeechStart: () => {
      log('speech start');
      vadIndicator.classList.add('speech');
    },
    onSpeechEnd: (audio) => {
      // audio is Float32Array @ 16 kHz
      lastSpeechAudio = audio;
      log(`speech end: ${audio.length} samples (~${(audio.length / 16000).toFixed(2)} s)`);
      vadIndicator.classList.remove('speech');
      playback.src = URL.createObjectURL(encodeWav(audio, 16000));
      playback.hidden = false;
    },
    onVADMisfire: () => log('VAD misfire (too short)'),
    onFrameProcessed: (probs, frame) => {
      frames++;
      const p = probs.isSpeech;
      vadVal.textContent = p.toFixed(2);
      vadBar.style.width = `${(p * 100).toFixed(0)}%`;
      // RMS for sanity
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      const rms = Math.sqrt(sum / frame.length);
      rmsVal.textContent = rms.toFixed(3);
      rmsBar.style.width = `${Math.min(100, rms * 400)}%`;
      const now = performance.now();
      if (now - lastRateCheck > 500) {
        const elapsed = (now - lastRateCheck) / 1000;
        frameRate.textContent = `${((frames - framesAtLastCheck) / elapsed) | 0}/s`;
        frameCount.textContent = frames;
        framesAtLastCheck = frames;
        lastRateCheck = now;
      }
      if (frames % 32 === 0) console.log('[VAD] p=', p.toFixed(3), 'rms=', rms.toFixed(3));
    },
  });

  try {
    await controller.start();
    setPill(permPill, 'granted', 'ok');
    setPill(vadPill, 'loaded', 'ok');
    log('MicVAD started — speak any time.');
  } catch (e) {
    setPill(vadPill, 'failed', 'bad');
    setPill(permPill, 'error', 'bad');
    log(`MicVAD start failed: ${e.message}`);
    console.error(e);
    controller = null;
  }
}

function stop() {
  if (!controller) return;
  controller.destroy();
  controller = null;
  setPill(ctxPill, 'closed', 'warn');
  log('stopped');
}

function recordTest() {
  if (!lastSpeechAudio) { log('no captured utterance yet — speak first'); return; }
  const wav = encodeWav(lastSpeechAudio, 16000);
  playback.src = URL.createObjectURL(wav);
  playback.hidden = false;
  playback.play().catch(() => {});
  log(`replaying last utterance: ${wav.size} bytes`);
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

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
recordBtn.addEventListener('click', recordTest);

log('ready. click "Start mic" to begin.');
