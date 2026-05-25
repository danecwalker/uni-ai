// Thin adapter around @ricky0123/vad-web (a battle-tested Silero VAD wrapper).
// Loaded globally as `vad` via the UMD bundle. We re-export a slim helper.

const VAD_BASE = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/';
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

class MicVadController {
  constructor(opts) {
    this.opts = opts;
    this.instance = null;
  }

  async start() {
    if (!window.vad) throw new Error('@ricky0123/vad-web not loaded');
    if (!this.instance) {
      this.instance = await window.vad.MicVAD.new({
        // tell the library where to fetch its assets from
        workletURL: VAD_BASE + 'vad.worklet.bundle.min.js',
        modelURL: VAD_BASE + 'silero_vad_v5.onnx',
        ortConfig: (ort) => { ort.env.wasm.wasmPaths = ORT_BASE; },
        model: 'v5',
        positiveSpeechThreshold: this.opts.positiveSpeechThreshold ?? 0.35,
        negativeSpeechThreshold: this.opts.negativeSpeechThreshold ?? 0.20,
        minSpeechFrames: this.opts.minSpeechFrames ?? 2,
        preSpeechPadFrames: this.opts.preSpeechPadFrames ?? 12,
        redemptionFrames: this.opts.redemptionFrames ?? 28, // ~900ms of trailing silence
        onSpeechStart: this.opts.onSpeechStart || (() => {}),
        onSpeechEnd: this.opts.onSpeechEnd || (() => {}),
        onVADMisfire: this.opts.onVADMisfire || (() => {}),
        onFrameProcessed: this.opts.onFrameProcessed || (() => {}),
      });
    }
    this.instance.start();
  }

  pause() { if (this.instance) this.instance.pause(); }
  destroy() {
    if (this.instance) {
      this.instance.destroy();
      this.instance = null;
    }
  }
}

window.MicVadController = MicVadController;
