// PredictPanel — contract layer.

import type { QuestionLegend, PredictPanelProps, QuestionDraft } from './PredictPanel.data';
import type { QuestionSpec } from '@/types/api';

export type { PredictPanelProps, QuestionDraft, QuestionLegend };

export interface PredictPanelEvents {
  onStateChange?: (text: string) => void;
  onQuestionsChange?: (questions: QuestionDraft[]) => void;
  /** Open the native dialog and load a document into the state field. */
  onLoadDocument?: () => void;
  /**
   * Run the request. `legend` maps each question id to the option content behind
   * the labels the answer will report, so the result can show the content too.
   */
  onSubmit?: (
    state: string,
    questions: Record<string, QuestionSpec>,
    legend: QuestionLegend,
  ) => void;
  onClear?: () => void;
}

export interface PredictPanelPublicApi {
  /** Append a blank question. */
  addQuestion: () => void;
}
