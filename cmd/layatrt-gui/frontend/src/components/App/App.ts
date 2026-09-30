// App — logic layer.
//
// Owns the page's state and calls the backend through the utils/backend domain.
// The view layer only renders what this returns.
//
// The interaction model: pick an engine with a native file dialog (or from the
// discovered list), write the questions yourself, load a document with a dialog,
// press Run. No JSON, no typed paths, no suggested schemas.

import { useCallback, useEffect, useState } from 'react';
import type {
  EngineFile,
  Limits,
  Metrics,
  PredictResponse,
  QuestionSpec,
  Status,
} from '@/types/api';
import {
  defaultEngineDir,
  discoverEngines,
  getMetrics,
  getStatus,
  hasBindings,
  limits as fetchLimits,
  loadEngine,
  pickDocumentFile,
  pickEngineFile,
  predict,
  readTextFile,
  revealEngine,
  unloadEngine,
} from '@/utils/backend';
import type { AppProps } from './App.data';
import { INITIAL_QUESTIONS, INITIAL_STATE_TEXT } from './App.data';
import type { EngineMode } from './EnginePanel/EnginePanel.data';
import { modeAcceptsPath } from './EnginePanel/EnginePanel.data';
import type { QuestionDraft, QuestionLegend } from './PredictPanel/PredictPanel.data';

export function useAppLogic(props: AppProps) {
  const [status, setStatus] = useState<Status | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [resultLegend, setResultLegend] = useState<QuestionLegend>({});
  const [engines, setEngines] = useState<EngineFile[]>([]);
  const [stateText, setStateText] = useState(INITIAL_STATE_TEXT);
  const [stateSource, setStateSource] = useState('');
  const [questions, setQuestions] = useState<QuestionDraft[]>(INITIAL_QUESTIONS);
  const [engineMode, setEngineMode] = useState<EngineMode>('auto');
  const [engineError, setEngineError] = useState<string | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [loadingEngine, setLoadingEngine] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [bindingsReady, setBindingsReady] = useState(true);

  const refreshStatus = useCallback(async () => {
    if (!hasBindings()) {
      setBindingsReady(false);
      return;
    }
    try {
      const [nextStatus, nextMetrics, nextLimits] = await Promise.all([
        getStatus(),
        getMetrics(),
        fetchLimits(),
      ]);
      setStatus(nextStatus);
      setMetrics(nextMetrics);
      setLimits(nextLimits);
      setBindingsReady(true);
    } catch (error) {
      setEngineError(message(error));
    }
  }, []);

  const rescan = useCallback(async () => {
    if (!hasBindings()) return;
    setScanning(true);
    try {
      setEngines(await discoverEngines());
    } catch (error) {
      setEngineError(message(error));
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    if (!hasBindings()) {
      setBindingsReady(false);
      return;
    }
    void refreshStatus();
    void rescan();
    // Only used to warm the dialog's starting folder.
    void defaultEngineDir().catch(() => undefined);
  }, [refreshStatus, rescan]);

  const handleLoadEngine = useCallback(
    async (selection: { path: string; contexts: number }, selectedMode: EngineMode = engineMode) => {
      if (!modeAcceptsPath(selectedMode, selection.path)) {
        setEngineError(selectedMode === 'tensorrt' ? 'TensorRT 需要 .engine 文件' : 'ONNX 引擎需要 .onnx 文件');
        return;
      }
      setLoadingEngine(true);
      setEngineError(null);
      try {
        const backend = selectedMode === 'auto' ? '' : selectedMode === 'tensorrt' ? 'tensorrt' : 'onnx';
        const provider = selectedMode.startsWith('onnx-') ? selectedMode.slice(5) : '';
        await loadEngine(selection.path, selection.contexts, backend, provider);
        setResult(null);
        setResultLegend({});
        await refreshStatus();
        await rescan();
      } catch (error) {
        setEngineError(message(error));
      } finally {
        setLoadingEngine(false);
      }
    },
    [engineMode, refreshStatus, rescan],
  );

  const handleModeChange = useCallback((nextMode: EngineMode) => {
    setEngineMode(nextMode);
    setEngineError(null);
    const current = status?.engine;
    if (current?.loaded && current.path) {
      if (modeAcceptsPath(nextMode, current.path)) {
        void handleLoadEngine({ path: current.path, contexts: current.contexts ?? 0 }, nextMode);
      } else {
        setEngineError(nextMode === 'tensorrt'
          ? '请选择 .engine 模型以切换到 TensorRT；当前模型仍在运行'
          : '请选择 .onnx 模型以切换到 ONNX；当前模型仍在运行');
      }
    }
  }, [handleLoadEngine, status?.engine]);

  const handleBrowseEngine = useCallback(async () => {
    try {
      const picked = await pickEngineFile();
      if (!picked) return; // cancelled
      await handleLoadEngine({ path: picked, contexts: 0 });
    } catch (error) {
      setEngineError(message(error));
    }
  }, [handleLoadEngine]);

  const handleUnload = useCallback(async () => {
    setLoadingEngine(true);
    try {
      await unloadEngine();
      setResult(null);
      setResultLegend({});
      await refreshStatus();
      await rescan();
    } catch (error) {
      setEngineError(message(error));
    } finally {
      setLoadingEngine(false);
    }
  }, [refreshStatus, rescan]);

  const handleReveal = useCallback(async (path: string) => {
    try {
      await revealEngine(path);
    } catch (error) {
      setEngineError(message(error));
    }
  }, []);

  const handleLoadDocument = useCallback(async () => {
    try {
      const picked = await pickDocumentFile();
      if (!picked) return; // cancelled
      setLoadingFile(true);
      const text = await readTextFile(picked);
      setStateText(text);
      setStateSource(picked);
      setResult(null);
      setResultLegend({});
      setPredictError(null);
    } catch (error) {
      setPredictError(message(error));
    } finally {
      setLoadingFile(false);
    }
  }, []);

  const handlePredict = useCallback(
    async (
      rawState: string,
      requestQuestions: Record<string, QuestionSpec>,
      legend: QuestionLegend,
    ) => {
      setPredicting(true);
      setPredictError(null);
      try {
        const response = await predict({ state: rawState, questions: requestQuestions });
        setResult(response);
        setResultLegend(legend);
        setMetrics(await getMetrics());
      } catch (error) {
        setPredictError(message(error));
      } finally {
        setPredicting(false);
      }
    },
    [],
  );

  const handleClear = useCallback(() => {
    setResult(null);
    setResultLegend({});
    setPredictError(null);
  }, []);

  return {
    version: props.version ?? status?.version ?? '—',
    status,
    metrics,
    limits,
    result,
    resultLegend,
    engines,
    engineMode,
    stateText,
    setStateText,
    stateSource,
    questions,
    setQuestions,
    engineError,
    predictError,
    loadingEngine,
    scanning,
    predicting,
    loadingFile,
    bindingsReady,
    refresh: refreshStatus,
    rescan,
    handleLoadEngine,
    handleModeChange,
    handleBrowseEngine,
    handleUnload,
    handleReveal,
    handleLoadDocument,
    handlePredict,
    handleClear,
  };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
