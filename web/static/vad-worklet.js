// AudioWorklet: produces 512-sample float32 frames at 16 kHz mono.
// If the AudioContext is already at 16 kHz, this is a pass-through.
// Otherwise we apply a one-pole low-pass anti-alias filter and then
// fractional decimation. Frame size 512 matches Silero VAD v5.

class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSr = 16000;
    this.targetFrame = 512;
    this.ratio = sampleRate / this.targetSr; // e.g. 3.0 at 48k, 6.0 at 96k, 1.0 at 16k
    this.buffer = new Float32Array(0);
    // Low-pass state (one-pole) — cutoff at ~7 kHz.
    this.lpY = 0;
    const cutoff = 7000;
    const dt = 1 / sampleRate;
    const rc = 1 / (2 * Math.PI * cutoff);
    this.alpha = dt / (rc + dt);
    this.readPos = 0; // fractional read pointer for resampling
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    let down;
    if (this.ratio === 1) {
      down = ch;
    } else {
      // 1-pole LPF in place (on a copy).
      const filt = new Float32Array(ch.length);
      let y = this.lpY;
      const a = this.alpha;
      for (let i = 0; i < ch.length; i++) {
        y = y + a * (ch[i] - y);
        filt[i] = y;
      }
      this.lpY = y;

      // Fractional decimation via linear interpolation.
      const outCount = Math.floor((filt.length - this.readPos) / this.ratio);
      down = new Float32Array(outCount);
      let pos = this.readPos;
      for (let i = 0; i < outCount; i++) {
        const idx = Math.floor(pos);
        const frac = pos - idx;
        const a0 = filt[idx];
        const a1 = idx + 1 < filt.length ? filt[idx + 1] : a0;
        down[i] = a0 + (a1 - a0) * frac;
        pos += this.ratio;
      }
      this.readPos = pos - filt.length; // carry remainder into next block
    }

    if (down.length > 0) {
      const combined = new Float32Array(this.buffer.length + down.length);
      combined.set(this.buffer);
      combined.set(down, this.buffer.length);
      this.buffer = combined;
    }

    while (this.buffer.length >= this.targetFrame) {
      const frame = this.buffer.slice(0, this.targetFrame);
      this.port.postMessage(frame, [frame.buffer]);
      this.buffer = this.buffer.slice(this.targetFrame);
    }
    return true;
  }
}

registerProcessor('vad-processor', VadProcessor);
