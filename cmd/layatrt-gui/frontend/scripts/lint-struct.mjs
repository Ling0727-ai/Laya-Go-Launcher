#!/usr/bin/env node
/**
 * lint:struct — enforce the frontend structure spec.
 *
 * Checks, under src/:
 *   1. no .js/.jsx source files (TypeScript only)
 *   2. every component directory has the four files, correctly named
 *   3. a component directory holds only its own files and child directories
 *   4. every utils/ subdirectory has an index.ts
 *   5. utils files are named for what they do, not "utils.ts"/"helpers.ts"
 *
 * Exits non-zero with a list of violations.
 */

import { readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const srcDir = join(root, 'src');

const problems = [];
const notes = [];

const rel = (p) => p.replace(root + '\\', '').replace(root + '/', '').replace(/\\/g, '/');

function isDir(p) {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function isFile(p) {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

/** Walk a directory tree, returning every file path. */
function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (isDir(full)) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

if (!isDir(srcDir)) {
  console.error('lint:struct: src/ not found');
  process.exit(1);
}

const allFiles = walk(srcDir);

// ── 1. TypeScript only ──────────────────────────────────────────────────────
for (const file of allFiles) {
  if (/\.(js|jsx)$/.test(file)) {
    problems.push(`${rel(file)}: use .ts/.tsx instead of .js/.jsx`);
  }
}

// ── 2 & 3. component directories ────────────────────────────────────────────
const componentsDir = join(srcDir, 'components');

function checkComponentDir(dir) {
  const name = basename(dir);
  const entries = readdirSync(dir);
  const files = entries.filter((e) => isFile(join(dir, e)));
  const dirs = entries.filter((e) => isDir(join(dir, e)));

  // Every non-component file in a component directory must be named after it.
  const expected = new Set([
    `${name}.tsx`,
    `${name}.data.ts`,
    `${name}.api.ts`,
    `${name}.ts`,
    'index.ts',
  ]);
  for (const file of files) {
    if (!expected.has(file)) {
      problems.push(
        `${rel(join(dir, file))}: a component directory may only hold ${name}.* files ` +
          `and child component directories`,
      );
    }
  }

  // The four-file split is required, except for a pure container that only
  // mounts children (no .ts/.data.ts of its own). We require all four and let
  // the author delete deliberately if a component is trivial.
  for (const required of [`${name}.tsx`, `${name}.data.ts`, `${name}.api.ts`, `${name}.ts`]) {
    if (!existsSync(join(dir, required))) {
      problems.push(`${rel(join(dir, required))}: missing (component "${name}" must be split into four files)`);
    }
  }

  // Child components live in their own subdirectories.
  for (const child of dirs) {
    const childPath = join(dir, child);
    if (!/^[A-Z]/.test(child)) {
      problems.push(`${rel(childPath)}: component subdirectory must be PascalCase`);
      continue;
    }
    checkComponentDir(childPath);
  }
}

if (isDir(componentsDir)) {
  for (const entry of readdirSync(componentsDir)) {
    const full = join(componentsDir, entry);
    if (!isDir(full)) {
      // index.ts is the only allowed flat file here.
      if (entry !== 'index.ts') {
        problems.push(`${rel(full)}: components/ may only contain component directories and index.ts`);
      }
      continue;
    }
    if (!/^[A-Z]/.test(entry)) {
      problems.push(`${rel(full)}: component directory must be PascalCase`);
      continue;
    }
    checkComponentDir(full);
  }
} else {
  notes.push('src/components/ does not exist yet');
}

// ── 4 & 5. utils ────────────────────────────────────────────────────────────
const utilsDir = join(srcDir, 'utils');
const BANNED_UTIL_NAMES = new Set(['utils.ts', 'helper.ts', 'helpers.ts', 'common.ts', 'misc.ts']);

if (isDir(utilsDir)) {
  const entries = readdirSync(utilsDir);
  const subdirs = entries.filter((e) => isDir(join(utilsDir, e)));
  const flatFiles = entries.filter((e) => isFile(join(utilsDir, e)));

  for (const file of flatFiles) {
    if (file === 'index.ts') continue;
    problems.push(
      `${rel(join(utilsDir, file))}: utils must be grouped by domain ` +
        `(e.g. utils/format/), not placed at the utils root`,
    );
  }

  if (subdirs.length === 0 && flatFiles.length > 0) {
    problems.push('src/utils/: create at least one domain subdirectory');
  }

  for (const sub of subdirs) {
    const subPath = join(utilsDir, sub);
    if (!existsSync(join(subPath, 'index.ts'))) {
      problems.push(`${rel(subPath)}/index.ts: missing (each utils domain must aggregate its exports)`);
    }
    for (const file of readdirSync(subPath)) {
      if (!isFile(join(subPath, file))) continue;
      if (BANNED_UTIL_NAMES.has(file)) {
        problems.push(`${rel(join(subPath, file))}: name the file for what it does, not "${file}"`);
      }
      if (!/\.ts$/.test(file)) {
        problems.push(`${rel(join(subPath, file))}: utils files must be .ts`);
      }
    }
  }
} else {
  notes.push('src/utils/ does not exist yet');
}

// ── report ──────────────────────────────────────────────────────────────────
if (notes.length) {
  for (const n of notes) console.log(`note: ${n}`);
}

if (problems.length) {
  console.error(`\nlint:struct found ${problems.length} problem(s):\n`);
  for (const p of problems) console.error(`  ✗ ${p}`);
  console.error('');
  process.exit(1);
}

console.log('lint:struct: structure is compliant');
