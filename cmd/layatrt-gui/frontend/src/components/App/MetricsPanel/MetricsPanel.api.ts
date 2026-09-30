// MetricsPanel — contract layer.

import type { MetricsPanelProps } from './MetricsPanel.data';

export type { MetricsPanelProps };

export interface MetricsPanelEvents {
  onRefresh?: () => void;
}

export interface MetricsPanelPublicApi {
  refresh: () => void;
}
