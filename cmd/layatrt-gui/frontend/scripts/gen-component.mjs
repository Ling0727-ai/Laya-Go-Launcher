#!/usr/bin/env node
/**
 * gen:comp — scaffold a component that satisfies the project structure spec.
 *
 *   npm run gen:comp Card
 *   npm run gen:comp Card/CardHeader
 *
 * Creates src/components/<path>/ with the four required files:
 *
 *   <Name>.tsx       view layer
 *   <Name>.data.ts   interfaces, constants, enums
 *   <Name>.api.ts    public contract (props, events, exposed functions)
 *   <Name>.ts        logic (state, effects, handlers)
 *
 * Nested paths put the child inside the parent's directory, which is what the
 * spec requires: a child component is never a sibling of its parent.
 */

import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const componentsDir = join(root, 'src', 'components');

const args = process.argv.slice(2);
const useVue = args.includes('--vue');
const target = args.find((a) => !a.startsWith('--'));

if (!target) {
  console.error('usage: npm run gen:comp <ComponentPath> [--vue]');
  console.error('  e.g. npm run gen:comp Card/CardHeader');
  process.exit(1);
}

if (useVue) {
  console.error('This project uses React (.tsx); the --vue flag is not supported here.');
  process.exit(1);
}

// Validate the path: every segment must be PascalCase, so the generated names
// line up with the directory names.
const segments = target.split('/').filter(Boolean);
for (const segment of segments) {
  if (!/^[A-Z][A-Za-z0-9]*$/.test(segment)) {
    console.error(`invalid component name "${segment}": use PascalCase (e.g. CardHeader)`);
    process.exit(1);
  }
}

const name = segments[segments.length - 1];
const dir = join(componentsDir, ...segments);

if (existsSync(dir)) {
  console.error(`already exists: ${dir}`);
  process.exit(1);
}
mkdirSync(dir, { recursive: true });

const files = {
  [`${name}.data.ts`]: `// ${name} — data layer: interfaces, constants, enums.
// No side effects, no I/O, no business logic.

export interface ${name}Props {
  /** Short label shown to the user. */
  title: string;
  /** Optional supporting text. */
  detail?: string;
}

export const ${name.toUpperCase()}_DEFAULT_TITLE = '${name}';
`,
  [`${name}.api.ts`]: `// ${name} — contract layer: what callers outside this component may rely on.
// Keep this stable; it is the only surface other modules should import.

import type { ${name}Props } from './${name}.data';

export type { ${name}Props };

export interface ${name}Events {
  /** Raised when the user activates the component. */
  onActivate?: (id: string) => void;
}

export interface ${name}PublicApi {
  /** Move keyboard focus into the component. */
  focus: () => void;
}
`,
  [`${name}.ts`]: `// ${name} — logic layer: state, derived values, handlers.
// The view imports from here; this file must not import from the view.

import { useCallback, useState } from 'react';
import type { ${name}Props, ${name}Events } from './${name}.data';
import { ${name.toUpperCase()}_DEFAULT_TITLE } from './${name}.data';

export function use${name}Logic(props: ${name}Props & ${name}Events) {
  const [isActive, setIsActive] = useState(false);

  const handleActivate = useCallback(() => {
    setIsActive((previous) => !previous);
    props.onActivate?.(props.title);
  }, [props]);

  return {
    isActive,
    handleActivate,
    title: props.title || ${name.toUpperCase()}_DEFAULT_TITLE,
    detail: props.detail,
  };
}
`,
  [`${name}.tsx`]: `// ${name} — view layer: markup and child component mounting only.

import type { FC } from 'react';
import type { ${name}Props, ${name}Events } from './${name}.data';
import { use${name}Logic } from './${name}';

export const ${name}: FC<${name}Props & ${name}Events> = (props) => {
  const { isActive, handleActivate, title, detail } = use${name}Logic(props);

  return (
    <div className={\`${name.toLowerCase()}\${isActive ? ' ${name.toLowerCase()}--active' : ''}\`}>
      <button type="button" onClick={handleActivate}>
        {title}
      </button>
      {detail ? <p>{detail}</p> : null}
    </div>
  );
};
`,
};

for (const [file, content] of Object.entries(files)) {
  writeFileSync(join(dir, file), content, 'utf8');
}

console.log(`created ${dir}`);
for (const file of Object.keys(files)) {
  console.log(`  ${file}`);
}
console.log('\nnext: fill in the logic in ' + `${name}.ts`, 'and the markup in ' + `${name}.tsx`);
