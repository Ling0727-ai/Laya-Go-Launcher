/**
 * Number formatting for probabilities, confidences and counts.
 *
 * The backend rounds to four decimals to match the Python SDK, so the UI keeps
 * that precision rather than inventing its own.
 */

/** Format a probability or confidence in [0,1] with four decimals. */
export function formatProbability(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toFixed(4);
}

/** Format a probability as a percentage, one decimal. */
export function formatPercent(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

/** Format a plain number with a fixed number of decimals. */
export function formatNumber(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toFixed(decimals);
}
