// AppHeader — data layer.

export interface AppHeaderProps {
  /** Application name shown on the left. */
  title: string;
  /** Version string from the backend. */
  version: string;
  /** Whether the TensorRT kernel came up. */
  kernelAvailable: boolean;
  /** TensorRT version, when available. */
  tensorrtVersion?: string;
  /** GPU name. */
  deviceName: string;
  /** "12.0" */
  computeCapability: string;
  /** Free VRAM in MiB. */
  vramFreeMB: number;
  /** Total VRAM in MiB. */
  vramTotalMB: number;
  /** Whether a model is loaded. */
  engineLoaded: boolean;
}

export const APP_HEADER_DEFAULT_TITLE = 'Laya Go Launcher';
