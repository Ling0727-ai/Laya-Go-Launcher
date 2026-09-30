/**
 * @fileoverview Formatting helpers for values shown in the UI.
 *
 * Usage:
 * ```ts
 * import { formatMs, formatProbability, formatPercent } from '@/utils/format';
 *
 * formatMs(4.4938);          // "4.49 ms"
 * formatProbability(0.99691); // "0.9969"
 * formatPercent(0.99691);     // "99.7%"
 * ```
 */

export { formatMs, formatDuration } from './duration';
export { formatProbability, formatPercent, formatNumber } from './number';
export { formatShape, formatDtype, formatTensor } from './tensor';
