// MetricsPanel — view layer.

import type { FC } from 'react';
import { formatMs, formatPercent } from '@/utils/format';
import type { MetricsPanelProps } from './MetricsPanel.data';
import type { MetricsPanelEvents } from './MetricsPanel.api';
import { useMetricsPanelLogic } from './MetricsPanel';

export const MetricsPanel: FC<MetricsPanelProps & MetricsPanelEvents> = (props) => {
  const { metrics, failureRate, percentiles, hasSamples } = useMetricsPanelLogic(props);

  return (
    <section className="panel metrics-panel">
      <div className="metrics-panel__head">
        <h2 className="panel__title">Metrics</h2>
        <button type="button" className="btn btn--ghost btn--tiny" onClick={() => props.onRefresh?.()}>
          Refresh
        </button>
      </div>

      <dl className="metrics-panel__counters">
        <div>
          <dt>requests</dt>
          <dd>{metrics.requests_total}</dd>
        </div>
        <div>
          <dt>predictions</dt>
          <dd>{metrics.predict_total}</dd>
        </div>
        <div>
          <dt>failures</dt>
          <dd className={metrics.requests_failed > 0 ? 'is-bad' : ''}>{metrics.requests_failed}</dd>
        </div>
        <div>
          <dt>failure rate</dt>
          <dd>{formatPercent(failureRate)}</dd>
        </div>
      </dl>

      <div className="metrics-panel__latency">
        <h3>Latency</h3>
        {hasSamples ? (
          <>
            <dl className="metrics-panel__percentiles">
              {percentiles.map((p) => (
                <div key={p.label}>
                  <dt>{p.label}</dt>
                  <dd className="mono">{formatMs(p.value)}</dd>
                </div>
              ))}
            </dl>
            <p className="metrics-panel__range">
              min {formatMs(metrics.latency_ms.min)} · max {formatMs(metrics.latency_ms.max)} ·{' '}
              {metrics.latency_samples} samples
            </p>
          </>
        ) : (
          <p className="metrics-panel__empty">No predictions yet.</p>
        )}
      </div>

      {metrics.last_run ? (
        <p className="metrics-panel__last">
          last run {metrics.last_run.at} · {formatMs(metrics.last_run.total_ms)} total ·{' '}
          {formatMs(metrics.last_run.inference_ms)} inference
        </p>
      ) : null}
    </section>
  );
};
