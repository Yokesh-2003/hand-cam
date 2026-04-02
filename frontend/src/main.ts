import './style.css'
import '@mediapipe/hands'
import '@mediapipe/camera_utils'

type Vec2 = { x: number; y: number }

type HandLandmark = { x: number; y: number; z?: number }
type HandsResults = {
  multiHandLandmarks?: HandLandmark[][]
}

const HandsCtor = (window as any).Hands as {
  new (cfg: { locateFile: (file: string) => string }): {
    setOptions(opts: Record<string, unknown>): void
    onResults(cb: (r: HandsResults) => void): void
    send(input: { image: HTMLVideoElement }): Promise<void>
  }
}

const HAND_CONNECTIONS = (window as any).HAND_CONNECTIONS as Array<[number, number]>
const CameraCtor = (window as any).Camera as {
  new (
    videoEl: HTMLVideoElement,
    cfg: { onFrame: () => Promise<void> | void; width: number; height: number },
  ): { start: () => Promise<void> }
}

const app = document.querySelector<HTMLDivElement>('#app')
if (!app) throw new Error('Missing #app')

app.innerHTML = `
  <div class="page">
    <header class="header">
      <div class="title">
        <h1>Hand Air Draw</h1>
        <p>Draw in the air by pointing your index finger. Hand bones are shown as lines.</p>
      </div>
      <div class="status">
        <div class="pill" id="trackingPill">Initializing…</div>
        <div class="pill pill-muted" id="backendPill">Python model: not connected</div>
      </div>
    </header>

    <main class="grid">
      <section class="stage">
        <div class="videoWrap">
          <video id="video" class="video" playsinline></video>
          <canvas id="overlay" class="overlay" aria-label="hand overlay"></canvas>
          <canvas id="draw" class="draw" aria-label="air drawing"></canvas>
        </div>
        <div class="hint" id="hint"></div>
      </section>

      <aside class="panel">
        <div class="card">
          <h2>Brush</h2>
          <div class="row">
            <label class="label" for="color">Color</label>
            <input id="color" type="color" value="#22c55e" />
          </div>
          <div class="row">
            <label class="label" for="size">Size</label>
            <input id="size" type="range" min="2" max="28" value="8" />
            <span class="mono" id="sizeLabel">8</span>
          </div>
          <div class="row">
            <button id="clear" class="btn">Clear</button>
            <button id="pause" class="btn btn-secondary">Pause</button>
          </div>
        </div>

        <div class="card">
          <h2>Gesture</h2>
          <div class="kv">
            <div class="k">Local “pointing”</div>
            <div class="v mono" id="localPointing">—</div>
          </div>
          <div class="kv">
            <div class="k">Python model \(p\)</div>
            <div class="v mono" id="pyProb">—</div>
          </div>
          <div class="kv">
            <div class="k">Drawing</div>
            <div class="v mono" id="drawingState">—</div>
          </div>
          <p class="small">
            If the backend is running, the page sends 21 hand landmarks and receives a neural-net probability.
            The app draws when either: local heuristic says “pointing”, or Python probability is high.
          </p>
        </div>
      </aside>
    </main>
  </div>
`

const els = {
  video: document.querySelector<HTMLVideoElement>('#video')!,
  overlay: document.querySelector<HTMLCanvasElement>('#overlay')!,
  draw: document.querySelector<HTMLCanvasElement>('#draw')!,
  hint: document.querySelector<HTMLDivElement>('#hint')!,
  trackingPill: document.querySelector<HTMLDivElement>('#trackingPill')!,
  backendPill: document.querySelector<HTMLDivElement>('#backendPill')!,
  color: document.querySelector<HTMLInputElement>('#color')!,
  size: document.querySelector<HTMLInputElement>('#size')!,
  sizeLabel: document.querySelector<HTMLSpanElement>('#sizeLabel')!,
  clear: document.querySelector<HTMLButtonElement>('#clear')!,
  pause: document.querySelector<HTMLButtonElement>('#pause')!,
  localPointing: document.querySelector<HTMLDivElement>('#localPointing')!,
  pyProb: document.querySelector<HTMLDivElement>('#pyProb')!,
  drawingState: document.querySelector<HTMLDivElement>('#drawingState')!,
}

const overlayCtx = els.overlay.getContext('2d', { alpha: true })!
const drawCtx = els.draw.getContext('2d', { alpha: true })!

let running = true
let camera: { start: () => Promise<void> } | null = null

let lastTip: Vec2 | null = null
let currentColor = els.color.value
let brushSize = Number(els.size.value)
els.sizeLabel.textContent = String(brushSize)

els.color.addEventListener('input', () => {
  currentColor = els.color.value
})
els.size.addEventListener('input', () => {
  brushSize = Number(els.size.value)
  els.sizeLabel.textContent = String(brushSize)
})
els.clear.addEventListener('click', () => {
  drawCtx.clearRect(0, 0, els.draw.width, els.draw.height)
})
els.pause.addEventListener('click', () => {
  running = !running
  els.pause.textContent = running ? 'Pause' : 'Resume'
  els.trackingPill.textContent = running ? 'Running' : 'Paused'
})

function resizeCanvasesToDisplay() {
  const rect = els.video.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return

  const dpr = window.devicePixelRatio || 1
  const w = Math.max(1, Math.round(rect.width * dpr))
  const h = Math.max(1, Math.round(rect.height * dpr))

  for (const c of [els.overlay, els.draw]) {
    if (c.width !== w) c.width = w
    if (c.height !== h) c.height = h
  }

  // Use CSS pixels for drawing commands.
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0)
  drawCtx.setTransform(dpr, 0, 0, dpr, 0, 0)
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height)
}

function drawSkeleton(landmarks: Array<{ x: number; y: number }>) {
  // canvas is scaled for DPR; draw in CSS pixels
  const rect = els.video.getBoundingClientRect()
  const w = rect.width
  const h = rect.height

  overlayCtx.save()
  overlayCtx.lineWidth = 3
  overlayCtx.strokeStyle = 'rgba(56, 189, 248, 0.9)'
  overlayCtx.beginPath()

  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = landmarks[a]
    const pb = landmarks[b]
    overlayCtx.moveTo(pa.x * w, pa.y * h)
    overlayCtx.lineTo(pb.x * w, pb.y * h)
  }

  overlayCtx.stroke()

  // index fingertip highlight (landmark 8)
  const tip = landmarks[8]
  overlayCtx.fillStyle = 'rgba(34, 197, 94, 0.9)'
  overlayCtx.beginPath()
  overlayCtx.arc(tip.x * w, tip.y * h, 7, 0, Math.PI * 2)
  overlayCtx.fill()
  overlayCtx.restore()
}

function dist(a: Vec2, b: Vec2) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.hypot(dx, dy)
}

// Simple “pointing” heuristic:
// - index extended: tip farther from wrist than pip
// - other fingers not extended too much (roughly)
function isPointingHeuristic(lm: Array<{ x: number; y: number }>) {
  const wrist = lm[0]
  const indexTip = lm[8]
  const indexPip = lm[6]

  const middleTip = lm[12]
  const middlePip = lm[10]
  const ringTip = lm[16]
  const ringPip = lm[14]
  const pinkyTip = lm[20]
  const pinkyPip = lm[18]

  const dIndexTip = dist(indexTip, wrist)
  const dIndexPip = dist(indexPip, wrist)
  const indexExtended = dIndexTip > dIndexPip + 0.02

  const middleExtended = dist(middleTip, wrist) > dist(middlePip, wrist) + 0.02
  const ringExtended = dist(ringTip, wrist) > dist(ringPip, wrist) + 0.02
  const pinkyExtended = dist(pinkyTip, wrist) > dist(pinkyPip, wrist) + 0.02

  const othersCurled = !middleExtended && !ringExtended && !pinkyExtended
  return indexExtended && othersCurled
}

function drawAirStroke(tipPx: Vec2) {
  drawCtx.save()
  drawCtx.lineCap = 'round'
  drawCtx.lineJoin = 'round'
  drawCtx.strokeStyle = currentColor
  drawCtx.lineWidth = brushSize

  if (!lastTip) {
    lastTip = tipPx
    drawCtx.restore()
    return
  }

  // ignore tiny jitter
  if (Math.hypot(tipPx.x - lastTip.x, tipPx.y - lastTip.y) < 1.25) {
    drawCtx.restore()
    return
  }

  drawCtx.beginPath()
  drawCtx.moveTo(lastTip.x, lastTip.y)
  drawCtx.lineTo(tipPx.x, tipPx.y)
  drawCtx.stroke()
  drawCtx.restore()
  lastTip = tipPx
}

let pyProb: number | null = null
let backendWs: WebSocket | null = null

function connectBackend() {
  try {
    backendWs = new WebSocket('ws://127.0.0.1:8000/ws')
  } catch {
    backendWs = null
    return
  }

  backendWs.onopen = () => {
    els.backendPill.textContent = 'Python model: connected'
    els.backendPill.classList.remove('pill-muted')
  }
  backendWs.onclose = () => {
    els.backendPill.textContent = 'Python model: not connected'
    els.backendPill.classList.add('pill-muted')
    pyProb = null
    window.setTimeout(connectBackend, 1500)
  }
  backendWs.onerror = () => {
    // onclose handles UI + retry
  }
  backendWs.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data)) as { p_pointing?: number }
      if (typeof msg.p_pointing === 'number') pyProb = msg.p_pointing
    } catch {
      // ignore
    }
  }
}

connectBackend()

function maybeSendToBackend(lm: Array<{ x: number; y: number; z?: number }>) {
  if (!backendWs || backendWs.readyState !== WebSocket.OPEN) return
  // send only x/y/z for 21 points
  backendWs.send(
    JSON.stringify({
      landmarks: lm.map((p) => [p.x, p.y, p.z ?? 0]),
    }),
  )
}

async function start() {
  els.trackingPill.textContent = 'Requesting camera…'

  const hands = new HandsCtor({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  })
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  })

  hands.onResults((results: HandsResults) => {
    if (!running) return

    resizeCanvasesToDisplay()
    clearOverlay()

    const lms = results.multiHandLandmarks?.[0]
    if (!lms) {
      els.hint.textContent = 'Show your hand to the camera.'
      els.trackingPill.textContent = 'No hand'
      lastTip = null
      els.localPointing.textContent = 'false'
      els.drawingState.textContent = 'idle'
      els.pyProb.textContent = pyProb == null ? '—' : pyProb.toFixed(3)
      return
    }

    els.trackingPill.textContent = 'Tracking'
    els.hint.textContent = 'Point index finger to draw.'

    drawSkeleton(lms)

    const localPointing = isPointingHeuristic(lms)
    els.localPointing.textContent = String(localPointing)

    maybeSendToBackend(lms)
    els.pyProb.textContent = pyProb == null ? '—' : pyProb.toFixed(3)

    const pySaysPointing = pyProb != null && pyProb > 0.65
    const shouldDraw = localPointing || pySaysPointing

    const rect = els.video.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    const tip = lms[8]
    const tipPx = { x: tip.x * w, y: tip.y * h }

    if (shouldDraw) {
      els.drawingState.textContent = 'drawing'
      drawAirStroke(tipPx)
    } else {
      els.drawingState.textContent = 'idle'
      lastTip = null
    }
  })

  camera = new CameraCtor(els.video, {
    onFrame: async () => {
      if (!running) return
      await hands.send({ image: els.video })
    },
    width: 1280,
    height: 720,
  })

  await camera.start()
  els.trackingPill.textContent = 'Running'
}

start().catch((err) => {
  console.error(err)
  els.trackingPill.textContent = 'Camera error'
  els.hint.textContent =
    'Could not start camera. Check browser permissions (HTTPS or localhost).'
})

els.video.addEventListener('loadedmetadata', () => {
  resizeCanvasesToDisplay()
})
window.addEventListener('resize', () => {
  resizeCanvasesToDisplay()
})
