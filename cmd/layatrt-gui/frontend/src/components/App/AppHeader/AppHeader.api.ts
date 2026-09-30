// AppHeader — contract layer.

import type { AppHeaderProps } from './AppHeader.data';

export type { AppHeaderProps };

export interface AppHeaderEvents {
  /** Raised when the user asks for a refresh. */
  onRefresh?: () => void;
}

export interface AppHeaderPublicApi {
  /** Re-render with fresh values. */
  set: (next: Partial<AppHeaderProps>) => void;
}
