// Backend access for the GUI.
//
// Every call goes through the Wails bindings, which are generated from
// internal/app.Bindings. Those bindings and the HTTP API call the same
// internal/inference use cases, so the window and a REST client cannot drift.
//
// The generated bindings live on window.go.app.Bindings — the namespace is the
// Go package name of the bound struct (internal/app), not "main".

import type {
  ConvertJob,
  EngineFile,
  EngineInfo,
  LauncherConfig,
  LoadState,
  Limits,
  Metrics,
  PredictResponse,
  Status,
} from '@/types/api';

interface Bindings {
  Status(): Promise<Status>;
  LoadEngine(path: string, contexts: number): Promise<EngineInfo>;
  LoadEngineWith(path: string, contexts: number, backend: string, provider: string): Promise<EngineInfo>;
  UnloadEngine(): Promise<{ unloaded: boolean }>;
  SetBudgets(maxLen: number, headMaxLen: number, padLen: number): Promise<Record<string, number>>;
  Predict(request: unknown): Promise<PredictResponse>;
  Tokenize(text: string, withSpecials: boolean): Promise<{
    count: number;
    ids: number[];
    tokens: string[];
    error?: string;
  }>;
  Metrics(): Promise<Metrics>;
  DefaultEngineDir(): Promise<string>;
  Limits(): Promise<Limits>;

  // Discovery and dialogs — the whole point of having a GUI.
  DiscoverEngines(): Promise<EngineFile[]>;
  PickEngineFile(): Promise<string>;
  PickTokenizerFile(): Promise<string>;
  PickDocumentFile(): Promise<string>;
  ReadTextFile(path: string): Promise<string>;
  RevealEngine(path: string): Promise<void>;

  // Launcher: fallback chain, config, conversion (shared with the server).
  APIAddress(): Promise<string>;
  LoadState(): Promise<LoadState>;
  AutoLoad(): Promise<LoadState>;
  GetConfig(): Promise<{ config: LauncherConfig; defaults: LauncherConfig; path?: string; tokenizer?: string }>;
  SaveConfig(cfg: LauncherConfig, reload: boolean): Promise<{ config: LauncherConfig; state?: LoadState; error?: string }>;
  StartConversion(seq: number, precision: string): Promise<ConvertJob>;
  ConversionJobs(): Promise<ConvertJob[]>;
  CancelConversion(): Promise<boolean>;
}

declare global {
  interface Window {
    go?: { app?: { Bindings?: Bindings } };
  }
}

/** True when the page is running inside the Wails window. */
export function hasBindings(): boolean {
  return typeof window !== 'undefined' && Boolean(window.go?.app?.Bindings);
}

function bindings(): Bindings {
  const b = window.go?.app?.Bindings;
  if (!b) {
    throw new Error(
      'Wails bindings are unavailable; this page is not running in the desktop app',
    );
  }
  return b;
}

export const getStatus = (): Promise<Status> => bindings().Status();

export const loadEngine = (path: string, contexts = 0, backend = '', provider = ''): Promise<EngineInfo> =>
  backend ? bindings().LoadEngineWith(path, contexts, backend, provider) : bindings().LoadEngine(path, contexts);

export const unloadEngine = (): Promise<{ unloaded: boolean }> => bindings().UnloadEngine();

export const setBudgets = (
  maxLen: number,
  headMaxLen: number,
  padLen: number,
): Promise<Record<string, number>> => bindings().SetBudgets(maxLen, headMaxLen, padLen);

export const predict = (request: unknown): Promise<PredictResponse> =>
  bindings().Predict(request);

export const tokenize = (text: string, withSpecials = false) =>
  bindings().Tokenize(text, withSpecials);

export const getMetrics = (): Promise<Metrics> => bindings().Metrics();

export const defaultEngineDir = (): Promise<string> => bindings().DefaultEngineDir();

/** Engines found in the usual locations. */
export const discoverEngines = (): Promise<EngineFile[]> => bindings().DiscoverEngines();

/** Native file dialog. Resolves to '' when the user cancels. */
export const pickEngineFile = (): Promise<string> => bindings().PickEngineFile();

export const pickTokenizerFile = (): Promise<string> => bindings().PickTokenizerFile();

/** Native file dialog for a document to analyse. */
export const pickDocumentFile = (): Promise<string> => bindings().PickDocumentFile();

/** Read a picked document as text. */
export const readTextFile = (path: string): Promise<string> => bindings().ReadTextFile(path);

/** What the loaded engine accepts, and what the service is using. */
export const limits = (): Promise<Limits> => bindings().Limits();

/** In-process REST address; the web console is served at its root. */
export const apiAddress = (): Promise<string> => bindings().APIAddress();

/** Most recent load with every fallback attempt. */
export const getLoadState = (): Promise<LoadState> => bindings().LoadState();

/** Run the fallback chain now. */
export const autoLoad = (): Promise<LoadState> => bindings().AutoLoad();

export const getConfig = () => bindings().GetConfig();

export const saveConfig = (cfg: LauncherConfig, reload: boolean) => bindings().SaveConfig(cfg, reload);

export const startConversion = (seq: number, precision: string): Promise<ConvertJob> =>
  bindings().StartConversion(seq, precision);

export const conversionJobs = (): Promise<ConvertJob[]> => bindings().ConversionJobs();

export const cancelConversion = (): Promise<boolean> => bindings().CancelConversion();

/** Show an engine's folder in the OS file manager. */
export const revealEngine = (path: string): Promise<void> => bindings().RevealEngine(path);
