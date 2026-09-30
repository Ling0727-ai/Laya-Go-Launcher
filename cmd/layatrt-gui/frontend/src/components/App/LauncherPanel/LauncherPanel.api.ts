// LauncherPanel — contract layer.

import type { LauncherPanelProps } from './LauncherPanel.data';

export type { LauncherPanelProps };

export interface LauncherPanelEvents {
  /** Raised after the panel (re)loaded a model, so the parent refreshes status. */
  onModelChanged?: () => void;
}
