// QuestionEditor — data layer.
//
// The draft shape lives with the parent (PredictPanel.data.ts) because the
// parent owns the list; this file re-exports it so the child's imports stay
// local, and adds the editor's own constants.

import type { QuestionDraft, QuestionOption } from '../PredictPanel.data';
import type { QuestionType } from '@/types/api';

export type { QuestionDraft, QuestionOption, QuestionType };

export interface QuestionEditorProps {
  draft: QuestionDraft;
  index: number;
  /** Whether the draft is complete enough to send. */
  valid: boolean;
  /** How many options the engine can score at once; 0 = unknown/unlimited. */
  markerCapacity: number;
}

export const QUESTION_TYPE_OPTIONS: Array<{
  value: QuestionType;
  label: string;
  hint: string;
}> = [
  { value: 'choice', label: 'choice', hint: '从若干选项里选一个，返回标签 A/B/C 和概率' },
  { value: 'score', label: 'score', hint: '有序等级，从上到下由低到高，返回期望值' },
  { value: 'noul', label: 'noul', hint: '校准过的真假概率，不需要选项' },
];

export const QUESTION_EDITOR_MAX_ID = 64;
