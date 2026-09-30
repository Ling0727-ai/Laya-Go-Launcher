// LauncherPanel — data layer.

import type { ConvertJob, LauncherConfig, LoadState } from '@/types/api';

export interface LauncherPanelProps {
  /** Bumped by the parent after a manual load, so the panel re-reads state. */
  refreshKey?: number;
}

export interface LauncherView {
  state: LoadState | null;
  config: LauncherConfig | null;
  jobs: ConvertJob[];
  apiAddr: string;
  busy: boolean;
  error: string | null;
  notice: string | null;
}

/** Fallback steps, in the default order, with display labels. */
export const STEP_LABELS: Record<string, string> = {
  tensorrt: 'TensorRT',
  'onnx-cuda': 'ONNX · CUDA',
  'onnx-directml': 'ONNX · DirectML',
  'onnx-cpu': 'ONNX · CPU',
};

export const ALL_STEPS = Object.keys(STEP_LABELS);

/** Context presets offered in the selector. 8192 is the default model. */
export const SEQ_CHOICES = [512, 1024, 2048, 4096, 8192] as const;

/** How often load/conversion state is polled while something is running (ms). */
export const POLL_BUSY_MS = 1500;
/** Idle polling interval (ms). */
export const POLL_IDLE_MS = 8000;

export const PHASE_LABELS: Record<string, string> = {
  idle: '未加载',
  loading: '加载中',
  ready: '就绪',
  failed: '失败',
};
