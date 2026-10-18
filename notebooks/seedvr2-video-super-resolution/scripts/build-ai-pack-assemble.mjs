import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFileSync, chmodSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import AdmZip from 'adm-zip'
import { matchesReleaseProvenance, payloadPins, modelNotice } from './ai-pack-pins.mjs'

// Assembly, not a new converter: all inputs are explicitly qualified build artifacts.
const root = resolve(import.meta.dirname, '..')
const options = { out: join(root, 'dist', 'seedvr2-openvino-pack'), archive: join(root, 'release', 'seedvr2-openvino-pack.zip') }
const keys = ['runtime', 'worker', 'models', 'complements', 'rife', 'licenses', 'provenance', 'out', 'archive']
const seen = new Set()
function fail(message) { throw new Error('AI pack build failed: ' + message) }
for (let i = 2; i < process.argv.length; ++i) {
  const flag = process.argv[i]
  if (flag === '--help') {
    console.log('Usage: build-ai-pack-assemble.mjs --runtime DIR --worker ELF --models DIR --complements DIR --rife DIR --licenses DIR --provenance JSON [--out DIR] [--archive FILE]')
    console.log('Linux x64 qualified-payload assembly; runtime DIR contains lib/. No download, Python or model inference.')
    process.exit(0)
  }
  const key = flag.slice(2)
  if (!flag.startsWith('--') || !keys.includes(key) || seen.has(key) || !process.argv[i + 1]) fail('unknown/duplicate/missing option ' + flag)
  seen.add(key)
  options[key] = resolve(process.argv[++i])
}
for (const key of keys.slice(0, 7)) if (!options[key]) fail('missing --' + key)
if (process.platform !== 'linux' || process.arch !== 'x64') fail('only Linux x64 is qualified')
if (!options.out.startsWith(join(root, 'dist') + '/')) fail('--out must be a generated child of dist/')
if (existsSync(options.out) || existsSync(options.archive)) fail('output/archive exists; refusing overwrite')
function regular(path, limit = 1024 ** 3) {
  const stat = lstatSync(path)
  if (!stat.isFile() || !Number.isSafeInteger(stat.size) || stat.size < 1 || stat.size > limit) fail('nonregular/empty/oversized input ' + path)
  return stat
}
function digest(path) { regular(path); return createHash('sha256').update(readFileSync(path)).digest('hex') }
function copy(source, destination) { regular(source); mkdirSync(dirname(destination), { recursive: true }); copyFileSync(source, destination) }
const revision = '09ced71023636e9bc8cdf9cdecfb2625d1e691e8'
const artifacts = [
  { path: 'seedvr2_ema_3b_fp16.safetensors', bytes: 6783018808, sha256: '2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304' },
  { path: 'ema_vae_fp16.safetensors', bytes: 501324814, sha256: '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1' }
].map((item) => ({ ...item, url: 'https://huggingface.co/numz/SeedVR2_comfyUI/resolve/' + revision + '/' + item.path + '?download=true' }))
regular(options.provenance, 1024 ** 2)
const provenance = JSON.parse(readFileSync(options.provenance, 'utf8'))
// Provenance is a release review record, NEVER native hash authority.
if (!matchesReleaseProvenance(provenance) || provenance.seedvr?.revision !== revision) fail('fixed release provenance differs')
function checkPin(path, pin) {
  if (regular(path).size !== pin.bytes || digest(path) !== pin.sha256) fail('fixed payload pin differs: ' + path)
}
for (const [name, pin] of Object.entries(payloadPins)) {
  const source = name.startsWith('lib/') ? join(options.runtime, name) :
    name.startsWith('rife.') ? join(options.rife, name) : join(options.licenses, name)
  checkPin(source, pin)
}
regular(options.worker, 64 * 1024 ** 2)
const env = { ...process.env, LD_LIBRARY_PATH: join(options.runtime, 'lib'), OPENVINO_LIB_PATH: join(options.runtime, 'lib') }
execFileSync(options.worker, ['--verify-models', '--models', options.models, '--complements', options.complements, '--structural-only'], { stdio: ['ignore', 'ignore', 'inherit'], env })
mkdirSync(dirname(options.out), { recursive: true })
mkdirSync(options.out, { mode: 0o755 })
copy(options.worker, join(options.out, 'seedvr2-openvino.bin'))
chmodSync(join(options.out, 'seedvr2-openvino.bin'), 0o755)
const launcher = ['#!/bin/sh', 'set -e', 'pack_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)', 'export OPENVINO_LIB_PATH="$pack_root/lib"', 'export LD_LIBRARY_PATH="$pack_root/lib:$LD_LIBRARY_PATH"', 'exec "$pack_root/seedvr2-openvino.bin" "$@"', ''].join('\n')
writeFileSync(join(options.out, 'seedvr2-openvino'), launcher, { mode: 0o755 })
const components = ['encoder-dynamic-t-hw', 'decoder-dynamic-t-hw', 'prelude_dynamic_fp16', 'postlude_dynamic_fp16']
const blocks = Array.from({ length: 32 }, (_, n) => 'block' + n + '_' + (n % 2 ? 'odd' : 'even') + '_fused_loop_tail_fp16')
const xml = [...blocks, ...[0, 1, 2, 10, 11, 31].map((n) => 'class' + n), ...components].map((name) => name + '.xml')
for (const name of [...xml, 'text_positive.f32']) copy(join(options.models, name), join(options.out, 'seedvr', name))
for (const name of [...Array.from({ length: 32 }, (_, n) => 'block' + String(n).padStart(2, '0')), ...components]) copy(join(options.complements, name + '.complement'), join(options.out, 'seedvr-complements', name + '.complement'))
for (const name of ['rife.xml', 'rife.bin']) copy(join(options.rife, name), join(options.out, name))
const libraries = Object.keys(payloadPins).filter((name) => name.startsWith('lib/lib')).map((name) => name.slice(4))
copy(join(options.runtime, 'lib', 'cache.json'), join(options.out, 'lib', 'cache.json'))
for (const name of libraries) copy(join(options.runtime, 'lib', name), join(options.out, 'lib', name))
for (const prefix of ['libopenvino.so', 'libopenvino_ir_frontend.so', 'libopenvino_intel_cpu_plugin.so', 'libopenvino_intel_gpu_plugin.so', 'libtbb.so', 'libcrypto.so']) if (!libraries.some((name) => name.startsWith(prefix))) fail('missing runtime library ' + prefix)
for (const name of readdirSync(options.licenses)) {
  if (!/^(LICENSE|NOTICE)[A-Z0-9_.-]*\.txt$/.test(name)) fail('unexpected attribution filename')
  copy(join(options.licenses, name), join(options.out, name))
}
for (const name of ['LICENSE-SEEDVR2.txt', 'LICENSE-RIFE-MIT.txt', 'LICENSE-OPENVINO.txt', 'LICENSE-OPENSSL.txt', 'LICENSE-ONE-TRACK-CAT.txt', 'NOTICE-AI-MODELS.txt']) if (!existsSync(join(options.out, name))) fail('missing attribution ' + name)
writeFileSync(join(options.out, 'NOTICE-SEEDVR2-RIFE.txt'), modelNotice + '\n')
writeFileSync(join(options.out, 'seedvr2-model-manifest.json'), JSON.stringify({ format: 1, modelId: 'seedvr2-3b-fp16', artifacts }, null, 2) + '\n')
function inventory(directory, prefix = '') {
  return readdirSync(directory).sort().flatMap((name) => {
    const path = join(directory, name), relative = prefix + name
    return lstatSync(path).isDirectory() ? inventory(path, relative + '/') : [{ path: relative, bytes: regular(path).size, sha256: digest(path) }]
  })
}
writeFileSync(join(options.out, 'manifest.json'), JSON.stringify({
  format: 1, model: 'seedvr2-3b-fp16-plus-rife-v4.25-v2',
  runtime: { name: 'OpenVINO Runtime', version: '2026.3.0', libraries }, provenance,
  prepared: { files: 79, derivedBinBytes: 7294499998, sourceBytes: 7284343622 }, files: inventory(options.out)
}, null, 2) + '\n')
const dependencies = execFileSync('ldd', [join(options.out, 'seedvr2-openvino.bin')], { encoding: 'utf8', env: { ...process.env, LD_LIBRARY_PATH: join(options.out, 'lib') } })
if (/not found/.test(dependencies)) fail('runtime dependency closure incomplete')
execFileSync(join(options.out, 'seedvr2-openvino'), ['--verify-models', '--models', join(options.out, 'seedvr'), '--complements', join(options.out, 'seedvr-complements'), '--structural-only'], { stdio: ['ignore', 'ignore', 'inherit'] })
const zip = new AdmZip()
zip.addLocalFolder(options.out, 'ai-pack')
const encoded = zip.toBuffer()
if (encoded.length >= 2 * 1024 ** 3) fail('archive exceeds 2GiB per-file limit')
mkdirSync(dirname(options.archive), { recursive: true })
writeFileSync(options.archive, encoded, { flag: 'wx' })
console.log('Built weightless native AI pack: ' + encoded.length + ' bytes')
