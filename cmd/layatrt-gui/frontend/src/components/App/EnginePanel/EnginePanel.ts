// EnginePanel — logic layer.

import { useCallback, useEffect, useState } from 'react';
import type { EnginePanelProps, EngineSelection } from './EnginePanel.data';
import { modeAcceptsPath } from './EnginePanel.data';
import type { EnginePanelEvents } from './EnginePanel.api';

export function useEnginePanelLogic(props: EnginePanelProps & EnginePanelEvents) {
  const [contexts, setContexts] = useState(0);
  const [chosen, setChosen] = useState<string>('');

  // Keep the highlighted row in step with what is actually loaded.
  useEffect(() => {
    if (props.engine.loaded && props.engine.path) {
      setChosen(props.engine.path);
    }
  }, [props.engine.loaded, props.engine.path]);

  const load = useCallback(
    (path: string) => {
      if (!path) return;
      setChosen(path);
      const selection: EngineSelection = { path, contexts };
      props.onLoad?.(selection);
    },
    [contexts, props],
  );

  const browse = useCallback(() => {
    props.onBrowse?.();
  }, [props]);

  // Split the scan into engines that can actually run this model and engines
  // that belong to some other project. Showing them together makes the user
  // guess which entry is theirs.
  const compatible = props.discovered.filter((e) => e.compatible && modeAcceptsPath(props.mode, e.path));
  const incompatible = props.discovered.filter((e) => !e.compatible);

  return {
    contexts,
    setContexts,
    chosen,
    load,
    browse,
    loaded: props.engine.loaded,
    inputs: props.engine.inputs ?? [],
    outputs: props.engine.outputs ?? [],
    activationMB: props.engine.activation_memory_mb,
    contextCount: props.engine.contexts,
    discovered: props.discovered,
    compatible,
    incompatible,
  };
}
