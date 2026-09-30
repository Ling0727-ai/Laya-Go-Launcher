// App — contract layer.

import type { AppProps, AppState } from './App.data';

export type { AppProps, AppState };

export interface AppEvents {
  /** Re-read status and metrics. */
  onRefresh?: () => void;
}

export interface AppPublicApi {
  /** Re-read everything from the backend. */
  refresh: () => Promise<void>;
}
