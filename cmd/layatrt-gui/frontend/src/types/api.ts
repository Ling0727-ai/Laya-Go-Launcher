/**
 * Global type declarations shared across the frontend.
 *
 * These mirror the JSON the Go transport emits. They live in one place so a
 * field rename is a compile error everywhere it is read.
 */

/** One engine input or output. */
export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
}

/** The loaded engine's description. */
export interface EngineInfo {
  loaded: boolean;
  path: string;
  backend: string;
  device: string;
  runtime: string;
  contexts: number;
  activation_memory_mb: number;
  inputs: TensorInfo[];
  outputs: TensorInfo[];
}

/** Kernel and device capability. */
export interface KernelInfo {
  available: boolean;
  tensorrt?: string;
  error?: string;
}

export interface DeviceInfo {
  name: string;
  compute_capability: string;
  vram_free_mb: number;
  vram_total_mb: number;
}

export interface Budgets {
  max_len: number;
  head_max_len: number;
  pad_len: number;
}

/** The header payload. */
export interface Status {
  version: string;
  uptime_s?: number;
  kernel: KernelInfo;
  device: DeviceInfo;
  engine: Partial<EngineInfo> & { loaded: boolean };
  budgets: Budgets;
}

/** The head's auxiliary action output. */
export interface ActionInfo {
  act_probability: number;
}

/** One answer. Only the fields matching the question type are populated. */
export interface Answer {
  type: 'choice' | 'score' | 'noul';
  choice?: string;
  probabilities?: Record<string, number>;
  confidence: number;
  action: ActionInfo;
  score?: number;
  legend?: Record<string, string>;
  noul?: number;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

export interface Timing {
  total_ms: number;
  tokenize_ms: number;
  inference_ms: number;
}

export interface PredictResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: Usage;
  timing: Timing;
}

/** The three decision primitives, as the UI names them. */
export type QuestionType = 'choice' | 'score' | 'noul';

/** A question as the API expects it. */
export interface QuestionSpec {
  type: QuestionType;
  instructions: string;
  criteria?: Record<string, string> | string[];
}

export interface LatencyStats {
  p50: number;
  p90: number;
  p99: number;
  min: number;
  max: number;
}

export interface Metrics {
  requests_total: number;
  requests_failed: number;
  predict_total: number;
  engine_loads: number;
  latency_samples: number;
  latency_ms: LatencyStats;
  last_run?: {
    at: string;
    total_ms: number;
    tokenize_ms: number;
    inference_ms: number;
  };
}

/** A tokenisation result. */
export interface TokenizeResult {
  count: number;
  ids: number[];
  tokens: string[];
  error?: string;
}

/** One discoverable engine on disk. */
export interface EngineFile {
  path: string;
  name: string;
  dir: string;
  size_mb: number;
  modified: string;
  in_use: boolean;
  /** False when the engine's IO tensors are not laya's (e.g. an image model). */
  compatible: boolean;
  /** Why an incompatible engine cannot be used. */
  reason?: string;
  /** Catalog metadata (seq ceiling, precision, format, tags). */
  seq_max?: number;
  precision?: string;
  format?: string;
  tags?: string;
}

/** What the loaded engine accepts, and what the service is using. */
export interface Limits {
  sequence_min: number;
  sequence_max: number;
  markers_max: number;
  fixed_length: boolean;
  /** The checkpoint's trained values, from rl_agent_config.json. */
  trained_max_len: number;
  trained_head_max_len: number;
  max_len: number;
  head_max_len: number;
  pad_len: number;
}

/** One step of an auto load, as it happened. */
export interface LoadAttempt {
  step: string;
  path: string;
  backend: string;
  provider: string;
  reason: string;
  ok: boolean;
  skipped?: boolean;
  error?: string;
  ms: number;
}

/** The launcher's most recent load. */
export interface LoadState {
  phase: 'idle' | 'loading' | 'ready' | 'failed' | string;
  mode: string;
  step?: string;
  path?: string;
  backend?: string;
  device?: string;
  model_config?: string;
  note?: string;
  error?: string;
  attempts: LoadAttempt[];
}

/** Process configuration (mirrors internal/config.Config). */
export interface LauncherConfig {
  engine_path: string;
  backend: string;
  provider: string;
  model_seq: number;
  fallback: string[];
  auto_convert: boolean;
  precision: string;
  contexts: number;
  engine_dir: string;
  http_addr: string;
  [key: string]: unknown;
}

/** A TensorRT conversion job. */
export interface ConvertJob {
  id: string;
  output: string;
  log: string;
  state: 'queued' | 'running' | 'done' | 'failed' | 'canceled' | string;
  error?: string;
  seconds: number;
  tail: string[];
}
