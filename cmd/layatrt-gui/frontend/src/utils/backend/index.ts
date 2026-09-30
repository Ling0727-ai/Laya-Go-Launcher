/**
 * @fileoverview Backend access for the GUI.
 *
 * Every call goes through the Wails bindings, which are generated from
 * internal/app.Bindings. Those bindings and the HTTP API call the same
 * internal/inference use cases, so the window and a REST client cannot drift.
 *
 * Usage:
 * ```ts
 * import { getStatus, loadEngine, predict } from '@/utils/backend';
 *
 * const status = await getStatus();
 * const result = await predict({ state, questions });
 * ```
 */

export {
  hasBindings,
  getStatus,
  loadEngine,
  unloadEngine,
  setBudgets,
  predict,
  tokenize,
  getMetrics,
  defaultEngineDir,
  discoverEngines,
  pickEngineFile,
  pickTokenizerFile,
  pickDocumentFile,
  readTextFile,
  limits,
  revealEngine,
  apiAddress,
  getLoadState,
  autoLoad,
  getConfig,
  saveConfig,
  startConversion,
  conversionJobs,
  cancelConversion,
} from './client';
