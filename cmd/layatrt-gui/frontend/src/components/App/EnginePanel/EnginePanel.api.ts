// EnginePanel — contract layer.

import type { EngineMode, EnginePanelProps, EngineSelection } from './EnginePanel.data';

export type { EnginePanelProps, EngineSelection };

export interface EnginePanelEvents {
  onModeChange?: (mode: EngineMode) => void;
  /** Load the engine at the given path. */
  onLoad?: (selection: EngineSelection) => void;
  /** Release the current engine. */
  onUnload?: () => void;
  /** Open the native file dialog and load whatever the user picks. */
  onBrowse?: () => void;
  /** Re-scan the usual locations for engines. */
  onRescan?: () => void;
  /** Show an engine's folder in the OS file manager. */
  onReveal?: (path: string) => void;
}

export interface EnginePanelPublicApi {
  /** Re-scan the usual locations. */
  rescan: () => void;
}
