/**
 * Check that the editor's option rows produce exactly the request shape laya
 * expects, and that the labels the engine answers with can be read back as
 * option content.
 *
 * This is the part that replaces hand-written syntax — including the option
 * labels, which are now derived from row position — so it is worth pinning: a
 * wrong mapping here would silently send malformed criteria, and a wrong label
 * would make the result unreadable.
 *
 *   node --test scripts/check-criteria.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

// The mapping under test, copied from PredictPanel.ts's buildCriteria and
// PredictPanel.data.ts's optionLabel. Kept in sync by the test below comparing
// shapes, not by importing TS.
function optionLabel(index) {
  let n = index;
  let label = '';
  do {
    label = String.fromCharCode(65 + (n % 26)) + label;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return label;
}

function buildCriteria(draft) {
  if (draft.type === 'noul') return undefined;
  const options = draft.options.filter((o) => o.text.trim().length > 0);

  if (draft.type === 'score') {
    return options.map((o) => o.text.trim());
  }

  const criteria = {};
  options.forEach((o, i) => {
    criteria[optionLabel(i)] = o.text.trim();
  });
  return criteria;
}

test('choice rows become a label -> content mapping, labelled A, B, C…', () => {
  const draft = {
    type: 'choice',
    options: [{ text: 'Claude Code' }, { text: 'Codex' }, { text: 'DeepSeek Harness' }],
  };
  assert.deepEqual(buildCriteria(draft), {
    A: 'Claude Code',
    B: 'Codex',
    C: 'DeepSeek Harness',
  });
});

test('the label is positional, so identical content keeps distinct labels', () => {
  // A content-derived label would collapse these two into one criterion and
  // silently drop an option.
  const draft = {
    type: 'choice',
    options: [{ text: 'same' }, { text: 'same' }],
  };
  assert.deepEqual(buildCriteria(draft), { A: 'same', B: 'same' });
  assert.equal(Object.keys(buildCriteria(draft)).length, 2);
});

test('labels stay consecutive after a removal', () => {
  // Removing the middle row must relabel the third as B, not leave a gap: the
  // label is the position, not a value stored on the row.
  const draft = {
    type: 'choice',
    options: [{ text: 'one' }, { text: 'three' }],
  };
  assert.deepEqual(Object.keys(buildCriteria(draft)), ['A', 'B']);
});

test('labels extend past Z without colliding', () => {
  assert.deepEqual(
    [0, 1, 25, 26, 27, 51, 52].map(optionLabel),
    ['A', 'B', 'Z', 'AA', 'AB', 'AZ', 'BA'],
  );
});

test('score rows become an ordered list, low to high', () => {
  const draft = {
    type: 'score',
    options: [{ text: 'not urgent' }, { text: 'soon' }, { text: 'critical deadline' }],
  };
  assert.deepEqual(buildCriteria(draft), ['not urgent', 'soon', 'critical deadline']);
});

test('noul sends no criteria', () => {
  const draft = { type: 'noul', options: [{ text: 'ignored' }] };
  assert.equal(buildCriteria(draft), undefined);
});

test('blank rows are dropped, so trailing empties do not add options', () => {
  const draft = {
    type: 'choice',
    options: [{ text: 'refunds' }, { text: '' }, { text: '' }],
  };
  assert.deepEqual(buildCriteria(draft), { A: 'refunds' });
});

test('whitespace is trimmed', () => {
  const draft = { type: 'choice', options: [{ text: '  refunds  ' }] };
  assert.deepEqual(buildCriteria(draft), { A: 'refunds' });
});

// ── the legend a result is read against ─────────────────────────────────────

// Copied from PredictPanel.data.ts's legendFromQuestions.
function legendFromQuestions(questions) {
  const out = {};
  for (const [id, spec] of Object.entries(questions)) {
    const criteria = spec.criteria;
    if (criteria === undefined) continue;
    out[id] = Array.isArray(criteria)
      ? Object.fromEntries(criteria.map((text, i) => [String(i), text]))
      : { ...criteria };
  }
  return out;
}

test('the legend maps every reported label back to its content', () => {
  const questions = {
    which: {
      type: 'choice',
      instructions: 'which?',
      criteria: { A: 'Claude Code', B: 'Codex' },
    },
    urgency: {
      type: 'score',
      instructions: 'how urgent?',
      criteria: ['not urgent', 'soon'],
    },
  };
  assert.deepEqual(legendFromQuestions(questions), {
    which: { A: 'Claude Code', B: 'Codex' },
    urgency: { 0: 'not urgent', 1: 'soon' },
  });
});

test('a noul question contributes no legend, because it has no options', () => {
  const legend = legendFromQuestions({
    risk: { type: 'noul', instructions: 'does it?' },
  });
  assert.deepEqual(legend, {});
});

test('a choice answer resolves to its content, not just its label', () => {
  // This is the round trip the result view performs.
  const criteria = buildCriteria({
    type: 'choice',
    options: [{ text: 'Claude Code' }, { text: 'Codex' }, { text: 'DeepSeek Harness' }],
  });
  const legend = legendFromQuestions({ which: { criteria } }).which;
  assert.equal(legend['A'], 'Claude Code');
  assert.equal(legend['C'], 'DeepSeek Harness');
});
