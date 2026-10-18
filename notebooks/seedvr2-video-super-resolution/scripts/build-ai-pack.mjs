import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'

// Build-only stdlib recipe; no Python/Torch is included in the native pack.
execFileSync(process.env.PYTHON ?? 'python3', [resolve(import.meta.dirname, 'prepare-ai-pack.py'), ...process.argv.slice(2)], { stdio: 'inherit' })
