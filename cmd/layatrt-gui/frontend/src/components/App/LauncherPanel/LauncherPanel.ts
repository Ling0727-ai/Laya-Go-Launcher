// LauncherPanel — logic layer. Polls the launcher and edits its base config.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { LauncherConfig } from '@/types/api';
import {
  apiAddress,
  autoLoad,
  cancelConversion,
  conversionJobs,
  getConfig,
  getLoadState,
  saveConfig,
  startConversion,
} from '@/utils/backend';
import type { LauncherPanelEvents } from './LauncherPanel.api';
import type { LauncherPanelProps, LauncherView } from './LauncherPanel.data';
import { ALL_STEPS, POLL_BUSY_MS, POLL_IDLE_MS } from './LauncherPanel.data';

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function useLauncherPanelLogic(props: LauncherPanelProps & LauncherPanelEvents) {
  const [view, setView] = useState<LauncherView>({
    state: null,
    config: null,
    jobs: [],
    apiAddr: '',
    busy: false,
    error: null,
    notice: null,
  });
  const [draft, setDraft] = useState<LauncherConfig | null>(null);
  const lastPhase = useRef<string>('');
  const onChanged = props.onModelChanged;

  const poll = useCallback(async () => {
    try {
      const [state, jobs] = await Promise.all([getLoadState(), conversionJobs()]);
      setView((v) => ({ ...v, state, jobs: jobs ?? [] }));
      if (lastPhase.current === 'loading' && state.phase !== 'loading') onChanged?.();
      lastPhase.current = state.phase;
      return state.phase === 'loading' || (jobs ?? []).some((j) => j.state === 'running');
    } catch (e) {
      setView((v) => ({ ...v, error: message(e) }));
      return false;
    }
  }, [onChanged]);

  // Initial config + address, and re-poll whenever the parent bumps refreshKey.
  useEffect(() => {
    void (async () => {
      try {
        const [c, addr] = await Promise.all([getConfig(), apiAddress()]);
        setView((v) => ({ ...v, config: c.config, apiAddr: addr }));
        setDraft(c.config);
      } catch (e) {
        setView((v) => ({ ...v, error: message(e) }));
      }
    })();
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const busy = await poll();
      if (alive) timer = setTimeout(tick, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [poll, props.refreshKey]);

  const run = useCallback(
    async (fn: () => Promise<string | null>) => {
      setView((v) => ({ ...v, busy: true, error: null, notice: null }));
      try {
        const notice = await fn();
        setView((v) => ({ ...v, notice }));
      } catch (e) {
        setView((v) => ({ ...v, error: message(e) }));
      } finally {
        setView((v) => ({ ...v, busy: false }));
        await poll();
        onChanged?.();
      }
    },
    [poll, onChanged],
  );

  const reload = () =>
    run(async () => {
      const st = await autoLoad();
      return st.phase === 'ready' ? `已加载 ${st.step}` : st.error ?? null;
    });

  const save = (andReload: boolean) =>
    run(async () => {
      if (!draft) return null;
      const r = await saveConfig(draft, andReload);
      setDraft(r.config);
      setView((v) => ({ ...v, config: r.config }));
      if (r.error) throw new Error(r.error);
      return andReload ? '已保存并按回退链重新加载' : '已保存（下次加载生效）';
    });

  const convert = () =>
    run(async () => {
      const job = await startConversion(draft?.model_seq ?? 8192, draft?.precision ?? 'fp16');
      return `开始构建 ${job.output.split(/[\\/]/).pop()}`;
    });

  const cancel = () => run(async () => ((await cancelConversion()) ? '已取消构建' : '没有正在运行的构建'));

  const toggleStep = (step: string) =>
    setDraft((d) => {
      if (!d) return d;
      const on = d.fallback.includes(step);
      const fallback = on ? d.fallback.filter((s) => s !== step) : [...d.fallback, step];
      return { ...d, fallback: fallback.length ? fallback : d.fallback };
    });

  const moveStep = (step: string, delta: number) =>
    setDraft((d) => {
      if (!d) return d;
      const f = [...d.fallback];
      const i = f.indexOf(step);
      const j = i + delta;
      if (i < 0 || j < 0 || j >= f.length) return d;
      [f[i], f[j]] = [f[j], f[i]];
      return { ...d, fallback: f };
    });

  const patch = (p: Partial<LauncherConfig>) => setDraft((d) => (d ? { ...d, ...p } : d));

  const orderedSteps = draft
    ? [...draft.fallback, ...ALL_STEPS.filter((s) => !draft.fallback.includes(s))]
    : ALL_STEPS;
  const dirty = !!draft && JSON.stringify(draft) !== JSON.stringify(view.config);
  const running = view.jobs.find((j) => j.state === 'running');

  return { view, draft, orderedSteps, dirty, running, reload, save, convert, cancel, toggleStep, moveStep, patch };
}
