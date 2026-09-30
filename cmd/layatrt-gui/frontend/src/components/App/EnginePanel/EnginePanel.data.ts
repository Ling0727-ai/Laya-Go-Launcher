// EnginePanel — data layer.

import type { EngineFile, EngineInfo } from '@/types/api';

export type EngineMode = 'auto' | 'tensorrt' | 'onnx-cuda' | 'onnx-directml' | 'onnx-cpu';

export const ENGINE_MODES: ReadonlyArray<{ value: EngineMode; label: string }> = [
  { value: 'auto', label: '自动' },
  { value: 'tensorrt', label: 'TensorRT' },
  { value: 'onnx-cuda', label: 'ONNX CUDA' },
  { value: 'onnx-directml', label: 'ONNX DirectML' },
  { value: 'onnx-cpu', label: 'ONNX CPU' },
];

export function modeAcceptsPath(mode: EngineMode, path: string): boolean {
  if (mode === 'auto') return true;
  return mode === 'tensorrt' ? /\.engine$/i.test(path) : /\.onnx$/i.test(path);
}

export interface EnginePanelProps {
  mode: EngineMode;
  engine: Partial<EngineInfo> & { loaded: boolean };
  /** Engines found on disk, offered as one-click choices. */
  discovered: EngineFile[];
  /** A load is in flight. */
  busy?: boolean;
  /** A discovery scan is in flight. */
  scanning?: boolean;
  /** Last error, if any. */
  error?: string | null;
}

export interface EngineSelection {
  path: string;
  contexts: number;
}

/** Suggested context counts. 0 asks the backend to size from free VRAM. */
export const CONTEXT_CHOICES = [0, 1, 2, 4, 8] as const;

/** How the loaded engine's path is shortened for display. */
export const ENGINE_PATH_TAIL = 3;
