// QuestionEditor — contract layer.

import type { QuestionDraft, QuestionEditorProps } from './QuestionEditor.data';

export type { QuestionDraft, QuestionEditorProps };

export interface QuestionEditorEvents {
  onChange?: (draft: QuestionDraft) => void;
  /** Remove this question, by its position in the list. */
  onRemove?: (index: number) => void;
}

export interface QuestionEditorPublicApi {
  focusId: () => void;
}
