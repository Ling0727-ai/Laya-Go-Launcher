/**
 * Tensor formatting for the engine panel.
 */

import type { TensorInfo } from '@/types/api';

/** Render a shape as "1x64". */
export function formatShape(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return 'scalar';
  return shape.join('×');
}

/** Human name for a dtype, with the byte width. */
export function formatDtype(dtype: string): string {
  const widths: Record<string, number> = {
    float32: 4,
    float16: 2,
    bfloat16: 2,
    int32: 4,
    int64: 8,
    bool: 1,
    int8: 1,
    uint8: 1,
  };
  const width = widths[dtype];
  return width ? `${dtype} (${width}B)` : dtype;
}

/** One-line summary of an IO tensor. */
export function formatTensor(tensor: TensorInfo): string {
  return `${tensor.name} · ${formatDtype(tensor.dtype)} · ${formatShape(tensor.shape)}`;
}
