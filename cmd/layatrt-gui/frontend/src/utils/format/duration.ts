/**
 * Duration formatting.
 *
 * Timings arrive in milliseconds from the Go side; the UI shows sub-millisecond
 * values with two decimals because that is the range the kernel works in.
 */

/** Format a millisecond value, choosing a sensible precision. */
export function formatMs(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  if (value < 1) return `${value.toFixed(3)} ms`;
  if (value < 100) return `${value.toFixed(2)} ms`;
  return `${value.toFixed(1)} ms`;
}

/** Format a second count as a compact uptime string. */
export function formatDuration(seconds: number | undefined | null): string {
  if (!seconds || seconds < 0) return '—';
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.floor(seconds % 60)}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}
