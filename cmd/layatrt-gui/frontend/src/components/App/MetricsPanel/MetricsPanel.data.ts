// MetricsPanel — data layer.

import type { Metrics } from '@/types/api';

export interface MetricsPanelProps {
  metrics: Metrics | null;
}

export const METRICS_EMPTY: Metrics = {
  requests_total: 0,
  requests_failed: 0,
  predict_total: 0,
  engine_loads: 0,
  latency_samples: 0,
  latency_ms: { p50: 0, p90: 0, p99: 0, min: 0, max: 0 },
};
