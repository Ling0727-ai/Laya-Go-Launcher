// PredictPanel — logic layer.
//
// Converts the editable rows into the request shape laya expects. This is the
// reason the UI models options as rows: the mapping from "a row with content" to
// `criteria` (a mapping for choice, a list for score) is mechanical, so the user
// never writes the syntax by hand — including the option labels, which are
// derived from row position.

import { useCallback, useMemo } from 'react';
import type { QuestionSpec } from '@/types/api';
import type { PredictPanelProps, QuestionDraft } from './PredictPanel.data';
import type { PredictPanelEvents } from './PredictPanel.api';
import {
  filledOptions,
  legendFromQuestions,
  nextQuestionId,
  optionCount,
  optionLabel,
} from './PredictPanel.data';

/** Build one question's `criteria` value from its option rows. */
export function buildCriteria(draft: QuestionDraft): QuestionSpec['criteria'] {
  if (draft.type === 'noul') return undefined;

  const options = filledOptions(draft);

  if (draft.type === 'score') {
    // A score question's criteria is a list, ordered low to high. Row order is
    // that order, and the answer's labels are its indices.
    return options.map((o) => o.text.trim());
  }

  // choice: a mapping of label -> content. The label is positional (A, B, …), so
  // it is unique even when two options carry the same text — which a
  // content-derived label would not be, and duplicate labels would collapse into
  // a single criterion.
  const criteria: Record<string, string> = {};
  options.forEach((o, i) => {
    criteria[optionLabel(i)] = o.text.trim();
  });
  return criteria;
}

/** True when a draft is complete enough to send. */
export function isDraftValid(draft: QuestionDraft): boolean {
  if (!draft.id.trim() || !draft.instructions.trim()) return false;
  if (draft.type === 'noul') return true;
  return filledOptions(draft).length >= 2;
}

/** Build the request's questions map. */
export function buildQuestions(drafts: QuestionDraft[]): Record<string, QuestionSpec> {
  const out: Record<string, QuestionSpec> = {};
  for (const draft of drafts) {
    if (!isDraftValid(draft)) continue;
    out[draft.id.trim()] = {
      type: draft.type,
      instructions: draft.instructions.trim(),
      criteria: buildCriteria(draft),
    };
  }
  return out;
}

export function usePredictPanelLogic(props: PredictPanelProps & PredictPanelEvents) {
  const questions = useMemo(() => buildQuestions(props.questions), [props.questions]);
  const validCount = Object.keys(questions).length;
  const invalidCount = props.questions.length - validCount;

  // Flag drafts that ask for more options than the engine can score, so the
  // problem shows up in the editor rather than as a failed request.
  const capacity = props.limits?.markers_max ?? 0;
  const overCapacity = useMemo(
    () =>
      capacity > 0
        ? props.questions.filter((q) => optionCount(q) > capacity).map((q) => q.id)
        : [],
    [props.questions, capacity],
  );

  const submit = useCallback(() => {
    if (!props.engineLoaded || validCount === 0) return;
    // Snapshot the option content behind every label before sending, so the
    // result can name what was chosen after the editor moves on.
    props.onSubmit?.(props.stateText, questions, legendFromQuestions(questions));
  }, [props, validCount, questions]);

  const addQuestion = useCallback(() => {
    props.onQuestionsChange?.([
      ...props.questions,
      {
        id: nextQuestionId(props.questions),
        type: 'choice',
        instructions: '',
        options: [{ text: '' }, { text: '' }],
      },
    ]);
  }, [props]);

  const removeQuestion = useCallback(
    (index: number) => {
      props.onQuestionsChange?.(props.questions.filter((_, i) => i !== index));
    },
    [props],
  );

  const updateQuestion = useCallback(
    (index: number, next: QuestionDraft) => {
      const copy = props.questions.slice();
      copy[index] = next;
      props.onQuestionsChange?.(copy);
    },
    [props],
  );

  return {
    questions,
    validCount,
    invalidCount,
    overCapacity,
    capacity,
    submit,
    addQuestion,
    removeQuestion,
    updateQuestion,
    canSubmit: props.engineLoaded && validCount > 0 && !props.busy,
    answers: props.result?.answers ?? {},
    timing: props.result?.timing,
    usage: props.result?.usage,
    model: props.result?.model,
  };
}
