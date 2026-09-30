// MetricsPanel — logic layer.

import { useMemo } from 'react';
import type { MetricsPanelProps } from './MetricsPanel.data';
import type { MetricsPanelEvents } from './MetricsPanel.api';
import { METRICS_EMPTY } from './MetricsPanel.data';

export function useMetricsPanelLogic(props: MetricsPanelProps & MetricsPanelEvents) {
  const metrics = props.metrics ?? METRICS_EMPTY;

  const failureRate = useMemo(() => {
    if (metrics.requests_total === 0) return 0;
    return metrics.requests_failed / metrics.requests_total;
  }, [metrics.requests_failed, metrics.requests_total]);

  const percentiles = useMemo(
    () => [
      { label: 'p50', value: metrics.latency_ms.p50 },
      { label: 'p90', value: metrics.latency_ms.p90 },
      { label: 'p99', value: metrics.latency_ms.p99 },
    ],
    [metrics.latency_ms],
  );

  return {
    metrics,
    failureRate,
    percentiles,
    hasSamples: metrics.latency_samples > 0,
  };
}
