// App — data layer.

import type { EngineFile, Limits, Metrics, PredictResponse, Status } from '@/types/api';
import type { QuestionDraft, QuestionLegend } from './PredictPanel/PredictPanel.data';

export interface AppProps {
  /** Injected for tests; production reads it from the backend. */
  version?: string;
}

export interface AppState {
  status: Status | null;
  metrics: Metrics | null;
  limits: Limits | null;
  result: PredictResponse | null;
  /** Option content behind the last result's labels, keyed by question id. */
  resultLegend: QuestionLegend;
  engines: EngineFile[];
  stateText: string;
  stateSource: string;
  questions: QuestionDraft[];
  engineError: string | null;
  predictError: string | null;
  loadingEngine: boolean;
  scanning: boolean;
  predicting: boolean;
  loadingFile: boolean;
}

/**
 * A window opens with one empty question and an empty document.
 *
 * This is a decision engine, so the user's own question is the starting point.
 * Nothing is pre-filled with a schema they did not ask for — the tool's job is
 * to run the question they write, not to suggest one.
 */
export const INITIAL_QUESTIONS: QuestionDraft[] = [
  {
    id: 'question_1',
    type: 'choice',
    instructions: '',
    options: [{ text: '' }, { text: '' }],
  },
];

export const INITIAL_STATE_TEXT = '';
