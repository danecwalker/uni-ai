// AI orb: particle sphere + halo + morphing ring + bloom.
// Ported from a reference scene to fit the BlobScene.setState/setAmp interface.

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

(function () {
  // Per-app-state presets. Map our richer state set onto the reference idle/listening/speaking triple.
  // colors are [r,g,b] in 0..1 — lerped per-frame.
  const STATE_PRESETS = {
    idle:       { amplitude: 0.30, frequency: 0.60, speed: 0.20, intensity: 1.6,
                  color1: [0.10, 0.20, 0.45], color2: [0.30, 0.20, 0.55] },
    armed:      { amplitude: 0.40, frequency: 0.80, speed: 0.45, intensity: 2.0,
                  color1: [0.00, 0.30, 1.00], color2: [0.25, 0.55, 1.00] },
    // user speaking → blue
    listening:  { amplitude: 0.55, frequency: 1.05, speed: 0.85, intensity: 2.4,
                  color1: [0.00, 0.40, 1.00], color2: [0.10, 0.70, 1.00] },
    thinking:   { amplitude: 0.70, frequency: 1.30, speed: 1.30, intensity: 2.6,
                  color1: [0.30, 0.15, 0.85], color2: [0.95, 0.30, 0.70] },
    // AI speaking → green
    talking:    { amplitude: 0.90, frequency: 1.50, speed: 1.50, intensity: 2.8,
                  color1: [0.00, 0.65, 0.30], color2: [0.35, 1.00, 0.55] },
    triggered:  { amplitude: 0.95, frequency: 1.55, speed: 1.40, intensity: 2.9,
                  color1: [0.10, 0.85, 0.45], color2: [0.60, 1.00, 0.60] },
  };

  // === GLSL ===
  const SIMPLEX = `
    vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
    vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
    vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
    vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
    float snoise(vec3 v){
      const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
      vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
      vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g;
      vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
      vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
      i=mod289(i);
      vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
      float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
      vec4 j=p-49.0*floor(p*ns.z*ns.z);
      vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
      vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy;
      vec4 h=1.0-abs(x)-abs(y);
      vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
      vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0;
      vec4 sh=-step(h,vec4(0.0));
      vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
      vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
      vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
      p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
      vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
      return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
    }
  `;

  const ORB_VERT = `
    ${SIMPLEX}
    uniform float uTime, uAmplitude, uFrequency;
    varying float vNoise;
    varying vec3 vLocalPos;
    varying vec3 vNormalView;
    void main() {
      vec3 pos = position;
      float noise = snoise(vec3(pos.x * uFrequency, pos.y * uFrequency, pos.z * uFrequency + uTime));
      vNoise = noise;
      pos += normal * noise * uAmplitude;
      vLocalPos = pos;
      vNormalView = normalize(normalMatrix * normal);
      vec4 mv = modelViewMatrix * vec4(pos, 1.0);
      gl_PointSize = (40.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }
  `;
  const ORB_FRAG = `
    precision highp float;
    uniform vec3 uColor1, uColor2;
    uniform float uIntensity;
    varying float vNoise;
    varying vec3 vLocalPos;
    varying vec3 vNormalView;
    void main() {
      vec2 cxy = 2.0 * gl_PointCoord - 1.0;
      float r = dot(cxy, cxy);
      if (r > 1.0) discard;
      float pAlpha = (1.0 - r) * 0.9;
      float facing = abs(vNormalView.z);
      float rim = pow(1.0 - facing, 4.0);
      float popOut = smoothstep(0.4, 0.9, vNoise) * 0.8;
      float frontFader = smoothstep(-0.1, 0.3, vNormalView.z);
      float finalAlpha = pAlpha * max(rim, popOut) * frontFader;
      float mixVal = smoothstep(-1.5, 1.5, -vLocalPos.x + vLocalPos.y);
      vec3 col = mix(uColor1, uColor2, mixVal);
      col *= uIntensity * (0.8 + popOut * 1.2 + rim * 0.8);
      gl_FragColor = vec4(col, finalAlpha);
    }
  `;

  const HALO_VERT = `
    varying vec3 vNormal;
    varying vec3 vPositionNormal;
    varying vec3 vLocalPos;
    void main() {
      vLocalPos = position;
      vNormal = normalize(normalMatrix * normal);
      vPositionNormal = normalize((modelViewMatrix * vec4(position, 1.0)).xyz);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  const HALO_FRAG = `
    precision highp float;
    uniform vec3 uColor1, uColor2;
    uniform float uIntensity;
    varying vec3 vNormal;
    varying vec3 vPositionNormal;
    varying vec3 vLocalPos;
    void main() {
      float intensity = pow(0.5 - dot(vNormal, vPositionNormal), 3.0);
      intensity = smoothstep(0.1, 0.9, intensity) * uIntensity;
      float mixVal = smoothstep(-2.0, 2.0, -vLocalPos.x + vLocalPos.y);
      vec3 col = mix(uColor1, uColor2, mixVal);
      gl_FragColor = vec4(col, intensity * 0.6);
    }
  `;

  const RING_VERT = `
    ${SIMPLEX}
    uniform float uTime, uMorphAmplitude, uMorphFrequency;
    varying vec3 vLocalPos;
    varying vec3 vNormalView;
    void main() {
      vec3 pos = position;
      float noise = snoise(vec3(pos.x * uMorphFrequency, pos.y * uMorphFrequency, uTime * 0.8));
      vec3 dir = normalize(vec3(pos.x, pos.y, 0.0));
      pos.z += noise * uMorphAmplitude * 1.5;
      pos.xy += dir.xy * noise * uMorphAmplitude * 0.5;
      vLocalPos = pos;
      vNormalView = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `;
  const RING_FRAG = `
    precision highp float;
    uniform vec3 uColor1, uColor2;
    uniform float uIntensity, uTime;
    varying vec3 vLocalPos;
    varying vec3 vNormalView;
    void main() {
      float angle = atan(vLocalPos.y, vLocalPos.x);
      float w1 = sin(angle * 2.0 + uTime * 2.0);
      float w2 = sin(angle * 3.0 - uTime * 1.5);
      float mixVal = (w1 + w2) * 0.25 + 0.5;
      vec3 col = mix(uColor1, uColor2, mixVal);
      float energy = pow(sin(angle - uTime * 4.0) * 0.5 + 0.5, 8.0);
      col += mix(uColor1, uColor2, 0.5) * energy * 1.2;
      float edgeAlpha = pow(abs(vNormalView.z), 4.0) * 0.8;
      gl_FragColor = vec4(col * uIntensity, edgeAlpha);
    }
  `;

  // === plumbing ===
  let renderer, scene, camera, composer, bloomPass;
  let particles, particleMat, halo, haloMat, ringGroup, glowRing, glowRingMat;
  const clonePreset = (p) => ({
    ...p,
    color1: [...p.color1],
    color2: [...p.color2],
  });
  let target = clonePreset(STATE_PRESETS.idle);
  let current = clonePreset(STATE_PRESETS.idle);
  let amp = 0, ampTarget = 0;
  let timeAccum = 0;
  let lastFrame = performance.now();

  const colorBlue = new THREE.Color(0x0044ff);
  const colorPink = new THREE.Color(0xff2266);

  function init(container) {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ReinhardToneMapping;
    renderer.toneMappingExposure = 1.6;
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = null;            // let the page bg show through
    renderer.setClearColor(0x000000, 0); // fully transparent clear

    camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
    camera.position.z = 9;

    // A. particle orb
    const particleGeometry = new THREE.SphereGeometry(1.8, 90, 90);
    particleMat = new THREE.ShaderMaterial({
      vertexShader: ORB_VERT, fragmentShader: ORB_FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uAmplitude: { value: current.amplitude },
        uFrequency: { value: current.frequency },
        uColor1: { value: colorBlue }, uColor2: { value: colorPink },
        uIntensity: { value: current.intensity },
      },
    });
    particles = new THREE.Points(particleGeometry, particleMat);
    scene.add(particles);

    // B. halo
    const haloGeometry = new THREE.SphereGeometry(2.1, 64, 64);
    haloMat = new THREE.ShaderMaterial({
      vertexShader: HALO_VERT, fragmentShader: HALO_FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.BackSide,
      uniforms: {
        uColor1: { value: colorBlue }, uColor2: { value: colorPink },
        uIntensity: { value: 1.0 },
      },
    });
    halo = new THREE.Mesh(haloGeometry, haloMat);
    scene.add(halo);

    // C. morphing ring
    ringGroup = new THREE.Group();
    glowRingMat = new THREE.ShaderMaterial({
      vertexShader: RING_VERT, fragmentShader: RING_FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: {
        uColor1: { value: colorBlue }, uColor2: { value: colorPink },
        uIntensity: { value: 0.6 }, uTime: { value: 0 },
        uMorphAmplitude: { value: current.amplitude },
        uMorphFrequency: { value: current.frequency },
      },
    });
    const ringGeom = new THREE.TorusGeometry(1.9, 0.25, 32, 200);
    glowRing = new THREE.Mesh(ringGeom, glowRingMat);
    ringGroup.add(glowRing);
    scene.add(ringGroup);

    // post: bloom
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    // (resolution, strength, radius, threshold) — strength is overridden per-frame.
    // Tight, high-threshold bloom: only the brightest pixels glow, glow doesn't spread far.
    bloomPass = new UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 0.25, 0.10, 0.85);
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());

    resize(container);
    window.addEventListener('resize', () => resize(container));
    animate();
  }

  function resize(container) {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    bloomPass.setSize(w, h);
  }

  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;

    // Lerp current → target.
    current.amplitude += (target.amplitude - current.amplitude) * 0.05;
    current.frequency += (target.frequency - current.frequency) * 0.05;
    current.speed     += (target.speed     - current.speed)     * 0.05;
    current.intensity += (target.intensity - current.intensity) * 0.05;
    for (let i = 0; i < 3; i++) {
      current.color1[i] += (target.color1[i] - current.color1[i]) * 0.06;
      current.color2[i] += (target.color2[i] - current.color2[i]) * 0.06;
    }
    amp = amp + (ampTarget - amp) * 0.22;
    particleMat.uniforms.uColor1.value.fromArray(current.color1);
    particleMat.uniforms.uColor2.value.fromArray(current.color2);
    haloMat.uniforms.uColor1.value.fromArray(current.color1);
    haloMat.uniforms.uColor2.value.fromArray(current.color2);
    glowRingMat.uniforms.uColor1.value.fromArray(current.color1);
    glowRingMat.uniforms.uColor2.value.fromArray(current.color2);

    // Speed-driven time accumulator (frame-rate independent).
    timeAccum += dt * current.speed * 1.4;
    const t = timeAccum;

    // Audio amplitude boosts visual intensity during 'talking'.
    const liveIntensity = current.intensity * (1.0 + amp * 0.6);
    const liveAmp = current.amplitude * (1.0 + amp * 0.4);

    particleMat.uniforms.uTime.value = t;
    particleMat.uniforms.uAmplitude.value = liveAmp;
    particleMat.uniforms.uFrequency.value = current.frequency;
    particleMat.uniforms.uIntensity.value = liveIntensity;

    haloMat.uniforms.uIntensity.value = liveIntensity * 0.8;

    glowRingMat.uniforms.uTime.value = t;
    glowRingMat.uniforms.uIntensity.value = liveIntensity * 1.0;
    glowRingMat.uniforms.uMorphAmplitude.value = liveAmp * 0.6;
    glowRingMat.uniforms.uMorphFrequency.value = current.frequency * 0.5;

    bloomPass.strength = liveIntensity * 0.05;

    particles.rotation.y = t * 0.15;
    particles.rotation.z = t * 0.05;
    ringGroup.rotation.x = Math.sin(t * 0.5) * 0.1;
    ringGroup.rotation.y = Math.cos(t * 0.3) * 0.1;

    composer.render();
  }

  window.BlobScene = {
    init(container) { init(container); },
    setState(state) {
      const preset = STATE_PRESETS[state];
      if (preset) target = clonePreset(preset);
    },
    setAmp(a) { ampTarget = Math.max(0, Math.min(1, a)); },
  };

  const mount = () => {
    const el = document.getElementById('blob');
    if (el && !el.dataset.mounted) {
      el.dataset.mounted = '1';
      init(el);
    }
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
