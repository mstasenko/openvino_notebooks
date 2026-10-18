import { spawnSync } from 'node:child_process'
import { resolve, join } from 'node:path'

// Native RIFE-only stream check, NOT Electron/FFmpeg/audio/spatial qualification.
const root = resolve(import.meta.dirname, '..', 'dist', 'seedvr2-openvino-pack')
const width = 64, height = 64, frames = 2, frameBytes = width * height * 3
const input = Buffer.alloc(frameBytes * frames)
for (let frame = 0; frame < frames; ++frame) {
  for (let y = 0; y < height; ++y) for (let x = 0; x < width; ++x) {
    const offset = frame * frameBytes + (y * width + x) * 3
    input[offset] = (x * 3 + frame * 17) % 256
    input[offset + 1] = (y * 3 + frame * 31) % 256
    input[offset + 2] = (x + y + frame * 47) % 256
  }
}
const result = spawnSync(join(root, 'seedvr2-openvino'), [
  '--rife', join(root, 'rife.xml'), '--width', String(width), '--height', String(height),
  '--fps', '30', '--output-width', String(width), '--output-height', String(height),
  '--output-fps', '60', '--duration', String(frames / 30), '--device', 'CPU'
], { input, maxBuffer: 1024 * 1024, env: { ...process.env, otc_CPU_ONLY: '1' } })
if (result.error || result.signal || result.status !== 0 || result.stdout.length !== frameBytes * 4) {
  process.stderr.write(result.stderr ?? '')
  throw new Error(`RIFE-only native smoke failed: status=${result.status} signal=${result.signal} bytes=${result.stdout?.length}`)
}
console.log('PASS native CPU RIFE-only: 2 RGB24 inputs →4 outputs, same64x64 geometry; no SeedVR models/sources')
const missing = spawnSync(join(root, 'seedvr2-openvino'), [
  '--width', '64', '--height', '64', '--fps', '30', '--output-width', '128',
  '--output-height', '128', '--output-fps', '30', '--duration', String(frames / 30), '--device', 'CPU'
], { input, maxBuffer: 1024 * 1024, env: { ...process.env, otc_CPU_ONLY: '1' } })
if (missing.error || missing.signal || missing.status === 0 || missing.stdout.length ||
    !missing.stderr.toString().includes('--models directory is required')) {
  throw new Error('Missing spatial models did not fail with the expected setup hint')
}
console.log('PASS missing SeedVR models rejected before inference with explicit --models setup hint')
