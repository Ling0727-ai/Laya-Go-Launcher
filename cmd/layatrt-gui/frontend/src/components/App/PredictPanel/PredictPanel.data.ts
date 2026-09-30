// PredictPanel — data layer.

import type { Answer, Limits, PredictResponse, QuestionSpec, QuestionType } from '@/types/api';

/**
 * One option of a question.
 *
 * `text` is the option's content: what the model reads and what the result
 * shows. A `choice` question answers with a positional label (A, B, …) derived
 * by `optionLabel`, so there is no label for the user to type.
 *
 * This mirrors laya's own `criteria` shape, which is a mapping for choice and a
 * list for score. Modelling it as rows means the user never types a separator or
 * has to remember the ordering rule.
 */
export interface QuestionOption {
  text: string;
}

/** One editable question. */
export interface QuestionDraft {
  /** Stable key for React lists; also the question id sent to the API. */
  id: string;
  type: QuestionType;
  instructions: string;
  options: QuestionOption[];
}

/**
 * The label a choice option is given, by position: A, B, … Z, AA, AB, …
 *
 * Derived rather than typed, because the label is the key the engine reports
 * probabilities under: it has to be unique even when two options share the same
 * content, which a positional label guarantees and a hand-typed one does not
 * (two identical hand-typed labels would collapse into one criterion).
 */
export function optionLabel(index: number): string {
  let n = index;
  let label = '';
  do {
    label = String.fromCharCode(65 + (n % 26)) + label;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return label;
}

/**
 * The option content behind each label the engine reports, per question id.
 *
 * The engine answers a `choice` question with the option's label and a `score`
 * question with its level index, so the label alone does not say what was
 * chosen. This is snapshotted when the request is built, so a result stays
 * readable even after the editor is changed.
 */
export type QuestionLegend = Record<string, Record<string, string>>;

export interface PredictPanelProps {
  /** Whether an engine is loaded; predicting without one is rejected. */
  engineLoaded: boolean;
  /** What the engine accepts, for validation and display. */
  limits: Limits | null;
  /** The last result, if any. */
  result?: PredictResponse | null;
  /** A request is in flight. */
  busy?: boolean;
  /** A file is being read. */
  loadingFile?: boolean;
  /** Last error. */
  error?: string | null;
  /** Free-form state text. */
  stateText: string;
  /** Name of the file the state came from, if any. */
  stateSource: string;
  /** Editable questions — the primary object in this panel. */
  questions: QuestionDraft[];
  /** Option content behind each reported label, keyed by question id. */
  questionLegend?: QuestionLegend;
}

/** A blank question. Two empty options so a choice question is usable at once. */
export function emptyQuestion(index: number): QuestionDraft {
  return {
    id: `question_${index}`,
    type: 'choice',
    instructions: '',
    options: [{ text: '' }, { text: '' }],
  };
}

/**
 * The next unused `question_N` id.
 *
 * Adding after a deletion must not reuse an id: two drafts with the same id
 * would collapse into a single entry in the request map, silently dropping a
 * question. The user may still rename a question afterwards.
 */
export function nextQuestionId(questions: QuestionDraft[]): string {
  const used = new Set(questions.map((q) => q.id.trim()));
  let n = questions.length + 1;
  while (used.has(`question_${n}`)) n += 1;
  return `question_${n}`;
}

/** The options that carry content, in order. */
export function filledOptions(draft: QuestionDraft): QuestionOption[] {
  return draft.options.filter((o) => o.text.trim().length > 0);
}

/** How many options a draft declares. noul always scores two ([false, true]). */
export function optionCount(draft: QuestionDraft): number {
  if (draft.type === 'noul') return 2;
  return filledOptions(draft).length;
}

/**
 * Snapshot the label → option content mapping out of a built request.
 *
 * A choice question's `criteria` already is that mapping (label → content) and a
 * score question's is the ordered level list, whose labels are its indices, so
 * the legend is read back from the request itself rather than recomputed.
 */
export function legendFromQuestions(
  questions: Record<string, QuestionSpec>,
): QuestionLegend {
  const out: QuestionLegend = {};
  for (const [id, spec] of Object.entries(questions)) {
    const criteria = spec.criteria;
    if (criteria === undefined) continue;
    out[id] = Array.isArray(criteria)
      ? Object.fromEntries(criteria.map((text, i) => [String(i), text]))
      : { ...criteria };
  }
  return out;
}

export type PredictAnswer = Answer;
