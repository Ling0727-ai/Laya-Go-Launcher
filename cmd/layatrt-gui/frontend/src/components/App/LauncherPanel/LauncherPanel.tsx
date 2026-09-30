// LauncherPanel — view layer. Fallback chain, last load, base config, conversion.

import type { FC } from 'react';
import type { LauncherPanelProps } from './LauncherPanel.data';
import type { LauncherPanelEvents } from './LauncherPanel.api';
import { PHASE_LABELS, SEQ_CHOICES, STEP_LABELS } from './LauncherPanel.data';
import { useLauncherPanelLogic } from './LauncherPanel';

const base = (p?: string) => (p ? p.split(/[\\/]/).pop() : '');

export const LauncherPanel: FC<LauncherPanelProps & LauncherPanelEvents> = (props) => {
  const l = useLauncherPanelLogic(props);
  const st = l.view.state;
  const phase = st?.phase ?? 'idle';

  return (
    <section className="panel launcher-panel" aria-labelledby="launcher-title">
      <div className="launcher-panel__head">
        <h2 id="launcher-title" className="panel__title">自动加载</h2>
        <span className={`launcher-panel__phase is-${phase}`}>{PHASE_LABELS[phase] ?? phase}</span>
        {st?.step ? <span className="launcher-panel__step">{STEP_LABELS[st.step] ?? st.step}</span> : null}
      </div>

      {st?.path ? (
        <p className="launcher-panel__model" title={st.path}>
          {base(st.path)}
          {st.device ? <span className="launcher-panel__dim"> · {st.device}</span> : null}
        </p>
      ) : null}

      <ol className="launcher-panel__attempts" aria-label="最近一次加载的尝试">
        {(st?.attempts ?? []).map((a, i) => (
          <li key={i} className={a.ok ? 'is-ok' : a.skipped ? 'is-skip' : 'is-bad'} title={a.path}>
            <span className="launcher-panel__badge">{STEP_LABELS[a.step] ?? a.step}</span>
            <span className="launcher-panel__file">{base(a.path) || '（无文件）'}</span>
            <span className="launcher-panel__why">
              {a.ok ? `${Math.round(a.ms)} ms` : a.skipped ? `跳过：${a.error}` : a.error}
            </span>
          </li>
        ))}
        {!st?.attempts?.length ? <li className="is-skip">尚未加载</li> : null}
      </ol>

      {l.draft ? (
        <div className="launcher-panel__form">
          <div className="launcher-panel__row">
            <label className="field field--narrow">
              <span className="field__label">目标上下文</span>
              <select
                className="field__input"
                value={l.draft.model_seq}
                onChange={(e) => l.patch({ model_seq: Number(e.target.value) })}
              >
                {SEQ_CHOICES.map((s) => (
                  <option key={s} value={s}>{s} tokens</option>
                ))}
              </select>
            </label>
            <label className="field field--narrow">
              <span className="field__label">TRT 精度</span>
              <select
                className="field__input"
                value={l.draft.precision}
                onChange={(e) => l.patch({ precision: e.target.value })}
              >
                <option value="fp16">fp16</option>
                <option value="fp32">fp32</option>
              </select>
            </label>
            <label className="launcher-panel__check">
              <input
                type="checkbox"
                checked={l.draft.auto_convert}
                onChange={(e) => l.patch({ auto_convert: e.target.checked })}
              />
              缺 TRT 引擎时自动转换
            </label>
          </div>

          <span className="field__label">回退链（勾选启用，箭头调整顺序）</span>
          <ul className="launcher-panel__chain">
            {l.orderedSteps.map((s) => {
              const on = l.draft!.fallback.includes(s);
              return (
                <li key={s} className={on ? '' : 'is-off'}>
                  <label>
                    <input type="checkbox" checked={on} onChange={() => l.toggleStep(s)} />
                    {STEP_LABELS[s]}
                  </label>
                  <button type="button" className="btn btn--ghost btn--tiny" aria-label={`${STEP_LABELS[s]} 上移`}
                    disabled={!on} onClick={() => l.moveStep(s, -1)}>↑</button>
                  <button type="button" className="btn btn--ghost btn--tiny" aria-label={`${STEP_LABELS[s]} 下移`}
                    disabled={!on} onClick={() => l.moveStep(s, 1)}>↓</button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}

      <div className="launcher-panel__actions">
        <button type="button" className="btn btn--primary" disabled={l.view.busy} onClick={() => void l.save(true)}>
          {l.dirty ? '保存并重新加载' : '按回退链重新加载'}
        </button>
        {l.dirty ? (
          <button type="button" className="btn btn--ghost" disabled={l.view.busy} onClick={() => void l.save(false)}>
            仅保存
          </button>
        ) : null}
        {l.running ? (
          <button type="button" className="btn btn--danger" onClick={() => void l.cancel()}>取消构建</button>
        ) : (
          <button type="button" className="btn btn--ghost" disabled={l.view.busy} onClick={() => void l.convert()}>
            构建 TRT 引擎
          </button>
        )}
      </div>

      {l.running ? (
        <div className="launcher-panel__build" aria-live="polite">
          <div className="launcher-panel__bar"><span /></div>
          <p>
            正在构建 {base(l.running.output)} · {Math.round(l.running.seconds)} s
          </p>
          <pre>{l.running.tail.slice(-3).join('\n')}</pre>
        </div>
      ) : null}

      {l.view.error ? <p className="launcher-panel__msg is-bad" role="alert">{l.view.error}</p> : null}
      {l.view.notice ? <p className="launcher-panel__msg is-ok">{l.view.notice}</p> : null}
      {l.view.apiAddr ? (
        <p className="launcher-panel__dim launcher-panel__api">
          HTTP API / 控制台：<code>http://{l.view.apiAddr}/</code>
        </p>
      ) : null}
    </section>
  );
};
