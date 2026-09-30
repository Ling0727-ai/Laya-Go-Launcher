// QuestionEditor — logic layer.

import { useCallback } from 'react';
import type { QuestionDraft, QuestionOption, QuestionEditorProps } from './QuestionEditor.data';
import type { QuestionEditorEvents } from './QuestionEditor.api';
import { QUESTION_EDITOR_MAX_ID } from './QuestionEditor.data';
import { filledOptions, optionCount, optionLabel } from '../PredictPanel.data';

export function useQuestionEditorLogic(props: QuestionEditorProps & QuestionEditorEvents) {
  const update = useCallback(
    (patch: Partial<QuestionDraft>) => {
      props.onChange?.({ ...props.draft, ...patch });
    },
    [props],
  );

  const setId = useCallback(
    (value: string) => update({ id: value.slice(0, QUESTION_EDITOR_MAX_ID) }),
    [update],
  );

  const setOption = useCallback(
    (index: number, patch: Partial<QuestionOption>) => {
      const options = props.draft.options.map((o, i) => (i === index ? { ...o, ...patch } : o));
      update({ options });
    },
    [props, update],
  );

  // A new row is blank; its label comes from its position, so nothing has to be
  // typed into it and the labels stay consecutive after a removal.
  const addOption = useCallback(() => {
    update({ options: [...props.draft.options, { text: '' }] });
  }, [props, update]);

  const removeOption = useCallback(
    (index: number) => {
      // Keep at least two rows: a choice question with fewer cannot be answered.
      if (props.draft.options.length <= 2) return;
      update({ options: props.draft.options.filter((_, i) => i !== index) });
    },
    [props, update],
  );

  // Switching to score keeps the option content; a score question's levels are
  // its rows in order, so no column is dropped.
  const setType = useCallback(
    (type: QuestionDraft['type']) => {
      if (type === props.draft.type) return;
      update({ type });
    },
    [props, update],
  );

  // Warn before the request fails: the engine can only score as many options as
  // its marker dimension allows.
  const options = optionCount(props.draft);
  const overCapacity = props.markerCapacity > 0 && options > props.markerCapacity;

  return {
    draft: props.draft,
    index: props.index,
    valid: props.valid,
    options,
    filled: filledOptions(props.draft).length,
    overCapacity,
    canRemoveOption: props.draft.options.length > 2,
    // The label each row will be reported under: a positional A/B/C for choice,
    // and the level's index for score.
    labelFor: (i: number) => (props.draft.type === 'score' ? String(i) : optionLabel(i)),
    update,
    setId,
    setType,
    setOption,
    addOption,
    removeOption,
    remove: () => props.onRemove?.(props.index),
  };
}
