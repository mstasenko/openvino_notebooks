import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { lstatSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import AdmZip from 'adm-zip'
import { matchesReleaseProvenance, payloadPins, modelNotice } from './ai-pack-pins.mjs'

const projectRoot = resolve(import.meta.dirname, '..')
const options = { root: join(projectRoot, 'dist', 'seedvr2-openvino-pack'), archive: join(projectRoot, 'release', 'seedvr2-openvino-pack.zip') }
const seen = new Set()
function fail(message) { throw new Error('AI pack verification failed: ' + message) }
for (let i = 2; i < process.argv.length; ++i) {
  if (process.argv[i] === '--help') { console.log('Usage: verify-ai-pack.mjs [--root DIR] [--archive FILE]'); process.exit(0) }
  const key = process.argv[i].slice(2)
  if (!process.argv[i].startsWith('--') || !['root', 'archive'].includes(key) || seen.has(key) || !process.argv[i + 1]) fail('unknown/duplicate/missing argument')
  seen.add(key)
  options[key] = resolve(process.argv[++i])
}
function regular(path, limit = 1024 ** 3) {
  const info = lstatSync(path)
  if (!info.isFile() || !Number.isSafeInteger(info.size) || info.size < 1 || info.size > limit) fail('invalid regular file ' + path)
  return info
}
function safe(path) {
  if (typeof path !== 'string' || !path || path.includes('\\') || path.split('/').some((part) => !part || part === '.' || part === '..') || path.startsWith('/')) fail('unsafe manifest path')
}
function hash(data) { return createHash('sha256').update(data).digest('hex') }
regular(join(options.root, 'manifest.json'), 1024 ** 2)
const manifest = JSON.parse(readFileSync(join(options.root, 'manifest.json'), 'utf8'))
if (manifest.format !== 1 || manifest.model !== 'seedvr2-3b-fp16-plus-rife-v4.25-v2' ||
    manifest.runtime?.name !== 'OpenVINO Runtime' || manifest.runtime?.version !== '2026.3.0' ||
    manifest.prepared?.files !== 79 || manifest.prepared?.derivedBinBytes !== 7294499998 ||
    manifest.prepared?.sourceBytes !== 7284343622) fail('unsupported pack/model/runtime')
if (!Array.isArray(manifest.files) || manifest.files.length < 90) fail('incomplete file inventory')
const names = new Set()
for (const file of manifest.files) {
  safe(file.path)
  if (names.has(file.path)) fail('duplicate inventory path')
  names.add(file.path)
  if (/\.(pth|onnx|safetensors)$|python|torch/i.test(file.path)) fail('build-only source/runtime leaked')
  if (file.path.startsWith('seedvr/') && file.path.endsWith('.bin')) fail('learned BIN leaked into small pack')
  const path = join(options.root, file.path)
  if (regular(path).size !== file.bytes || !/^[a-f0-9]{64}$/.test(file.sha256) || hash(readFileSync(path)) !== file.sha256) fail('file hash/size mismatch ' + file.path)
}
for (const name of ['seedvr2-openvino', 'seedvr2-openvino.bin', 'rife.xml', 'rife.bin', 'seedvr2-model-manifest.json',
  'LICENSE-SEEDVR2.txt', 'LICENSE-RIFE-MIT.txt', 'LICENSE-OPENVINO.txt', 'LICENSE-OPENSSL.txt', 'LICENSE-ONE-TRACK-CAT.txt', 'NOTICE-AI-MODELS.txt']) if (!names.has(name)) fail('required payload missing ' + name)
if ((regular(join(options.root, 'seedvr2-openvino')).mode & 0o111) === 0 ||
    (regular(join(options.root, 'seedvr2-openvino.bin')).mode & 0o111) === 0 ||
    readFileSync(join(options.root, 'seedvr2-openvino.bin')).subarray(0, 4).toString() !== '\u007fELF') fail('invalid native executable')
const recipe = JSON.parse(readFileSync(join(options.root, 'seedvr2-model-manifest.json'), 'utf8'))
const expected = [
  ['seedvr2_ema_3b_fp16.safetensors', 6783018808, '2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304'],
  ['ema_vae_fp16.safetensors', 501324814, '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1']
]
const revision = '09ced71023636e9bc8cdf9cdecfb2625d1e691e8'
if (recipe.format !== 1 || recipe.modelId !== 'seedvr2-3b-fp16' || recipe.artifacts?.length !== 2) fail('unsupported source recipe')
for (const [path, bytes, sha256] of expected) {
  const item = recipe.artifacts.find((entry) => entry.path === path)
  if (!item || item.bytes !== bytes || item.sha256 !== sha256 ||
      item.url !== 'https://huggingface.co/numz/SeedVR2_comfyUI/resolve/' + revision + '/' + path + '?download=true') fail('source pin changed')
}
const provenance = manifest.provenance
if (!matchesReleaseProvenance(provenance) || provenance.seedvr?.revision !== revision) fail('fixed release provenance differs')
for (const [name, pin] of Object.entries(payloadPins)) {
  if (!names.has(name) || regular(join(options.root, name)).size !== pin.bytes ||
      hash(readFileSync(join(options.root, name))) !== pin.sha256) fail('fixed payload pin differs: ' + name)
}
if (!names.has('NOTICE-SEEDVR2-RIFE.txt') ||
    readFileSync(join(options.root, 'NOTICE-SEEDVR2-RIFE.txt'), 'utf8') !== modelNotice + '\n') fail('fixed model attribution differs')

const env = { ...process.env, LD_LIBRARY_PATH: join(options.root, 'lib'), OPENVINO_LIB_PATH: join(options.root, 'lib') }
const dependencies = execFileSync('ldd', [join(options.root, 'seedvr2-openvino.bin')], { encoding: 'utf8', env })
if (/not found/.test(dependencies)) fail('dependency closure incomplete')
// Native compiled authority authenticates all42XML+text+36complements, without Core.
execFileSync(join(options.root, 'seedvr2-openvino'), ['--verify-models', '--models', join(options.root, 'seedvr'), '--complements', join(options.root, 'seedvr-complements'), '--structural-only'], { stdio: ['ignore', 'ignore', 'inherit'], env })
regular(options.archive, 2 * 1024 ** 3 - 1)
const zip = new AdmZip(options.archive)
const archiveNames = new Set()
for (const entry of zip.getEntries()) {
  safe(entry.entryName.endsWith('/') ? entry.entryName.slice(0, -1) : entry.entryName)
  if (archiveNames.has(entry.entryName) || !entry.entryName.startsWith('ai-pack/')) fail('unsafe/duplicate archive entry')
  archiveNames.add(entry.entryName)
  if (!entry.isDirectory && entry.entryName !== 'ai-pack/manifest.json' && !names.has(entry.entryName.slice(8))) fail('unmanifested archive file')
}
for (const file of manifest.files) {
  const entry = zip.getEntry('ai-pack/' + file.path)
  if (!entry || entry.header.size !== file.bytes) fail('archive size mismatch')
  const bytes = entry.getData()
  if (bytes.length !== file.bytes || hash(bytes) !== file.sha256) fail('archive hash mismatch')
}
if (JSON.stringify(JSON.parse(zip.readAsText('ai-pack/manifest.json'))) !== JSON.stringify(manifest)) fail('archive manifest differs')
for (const name of ['seedvr2-openvino', 'seedvr2-openvino.bin']) if ((zip.getEntry('ai-pack/' + name).header.fileAttr & 0o111) === 0) fail('archive executable permissions missing')
console.log('Verified native structural pack; no learned source/BIN download or inference performed')
