import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const projectRoot = path.resolve(__dirname, '..')
const srcDir = path.join(projectRoot, 'node_modules', '@mediapipe', 'hands')
const outDir = path.join(projectRoot, 'public', 'mediapipe', 'hands')

const exts = new Set(['.js', '.wasm', '.data', '.tflite', '.binarypb'])

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true })
}

function copyFlatFiles(fromDir, toDir) {
  ensureDir(toDir)
  const entries = fs.readdirSync(fromDir, { withFileTypes: true })
  for (const ent of entries) {
    if (!ent.isFile()) continue
    const ext = path.extname(ent.name).toLowerCase()
    if (!exts.has(ext)) continue
    fs.copyFileSync(path.join(fromDir, ent.name), path.join(toDir, ent.name))
  }
}

try {
  copyFlatFiles(srcDir, outDir)
  // eslint-disable-next-line no-console
  console.log(`[mediapipe] copied hands assets -> ${path.relative(projectRoot, outDir)}`)
} catch (err) {
  // eslint-disable-next-line no-console
  console.warn('[mediapipe] could not copy assets (is npm install done?)', err)
  process.exitCode = 0
}

