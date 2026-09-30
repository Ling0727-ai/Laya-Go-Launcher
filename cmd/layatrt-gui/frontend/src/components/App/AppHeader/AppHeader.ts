// AppHeader — logic layer.

import { useMemo } from 'react';
import type { AppHeaderProps } from './AppHeader.data';
import type { AppHeaderEvents } from './AppHeader.api';

export function useAppHeaderLogic(props: AppHeaderProps & AppHeaderEvents) {
  const vram = useMemo(() => {
    if (!props.vramTotalMB) return '—';
    const used = props.vramTotalMB - props.vramFreeMB;
    const pct = Math.round((used / props.vramTotalMB) * 100);
    return `${props.vramFreeMB} / ${props.vramTotalMB} MiB 可用（已用 ${pct}%）`;
  }, [props.vramFreeMB, props.vramTotalMB]);

  const kernelLabel = useMemo(
    () => (props.kernelAvailable ? `TensorRT ${props.tensorrtVersion ?? ''}`.trim() : '内核不可用'),
    [props.kernelAvailable, props.tensorrtVersion],
  );

  const deviceLabel = useMemo(
    () => `${props.deviceName || '无设备'} · sm_${props.computeCapability.replace('.', '')}`,
    [props.deviceName, props.computeCapability],
  );

  return {
    vram,
    kernelLabel,
    deviceLabel,
    healthy: props.kernelAvailable,
    engineLabel: props.engineLoaded ? '已加载' : '未加载',
  };
}
