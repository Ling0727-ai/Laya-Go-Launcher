// EnginePanel — view layer.
//
// The primary action is a native file dialog and a one-click list of engines
// found on disk. There is no path text field: typing a Windows path by hand is
// not something a GUI should ever ask for.

import type { FC } from 'react';
import type { EngineFile } from '@/types/api';
import { formatShape, formatDtype } from '@/utils/format';
import type { EnginePanelProps } from './EnginePanel.data';
import { CONTEXT_CHOICES, ENGINE_MODES } from './EnginePanel.data';
import type { EnginePanelEvents } from './EnginePanel.api';
import { useEnginePanelLogic } from './EnginePanel';

export const EnginePanel: FC<EnginePanelProps & EnginePanelEvents> = (props) => {
  const {
    contexts,
    setContexts,
    chosen,
    load,
    browse,
    loaded,
    inputs,
    outputs,
    activationMB,
    contextCount,
    compatible,
    incompatible,
  } = useEnginePanelLogic(props);

  return (
    <section className="panel engine-panel">
      <h2 className="panel__title">模型</h2>

      <div className="engine-panel__modes" role="group" aria-label="执行引擎">
        {ENGINE_MODES.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            className={`engine-panel__mode${props.mode === value ? ' engine-panel__mode--active' : ''}`}
            aria-pressed={props.mode === value}
            disabled={props.busy}
            onClick={() => props.onModeChange?.(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {loaded ? <p className="engine-panel__active">当前运行：{props.engine.backend} · {props.engine.device}</p> : null}

      <div className="engine-panel__actions">
        <button type="button" className="btn btn--primary" disabled={props.busy} onClick={browse}>
          {props.busy ? '加载中…' : '浏览并加载…'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={!loaded || props.busy}
          onClick={() => props.onUnload?.()}
        >
          卸载
        </button>
        <button
          type="button"
          className="btn btn--ghost"
          disabled={props.scanning}
          onClick={() => props.onRescan?.()}
        >
          {props.scanning ? '扫描中…' : '重新扫描'}
        </button>

        <label className="engine-panel__contexts" title="并行执行上下文数量；自动 = 按空闲显存决定">
          <span>并发</span>
          <select
            className="field__input"
            value={contexts}
            onChange={(event) => setContexts(Number(event.target.value))}
          >
            {CONTEXT_CHOICES.map((n) => (
              <option key={n} value={n}>
                {n === 0 ? '自动' : n}
              </option>
            ))}
          </select>
        </label>
      </div>

      {props.error ? <p className="engine-panel__error">{props.error}</p> : null}

      {compatible.length > 0 || incompatible.length > 0 ? (
        <div className="engine-panel__list">
          <h3 className="engine-panel__list-title">
            找到 {compatible.length} 个可用模型
            {incompatible.length > 0 ? `（另有 ${incompatible.length} 个不适用，已折叠）` : ''}
          </h3>
          <ul>
            {compatible.map((e: EngineFile) => {
              const active = e.path === chosen || e.in_use;
              return (
                <li
                  key={e.path}
                  className={`engine-row${active ? ' engine-row--active' : ''}`}
                >
                  <button
                    type="button"
                    className="engine-row__pick"
                    disabled={props.busy}
                    onClick={() => load(e.path)}
                    title={e.path}
                  >
                    <span className="engine-row__name">
                      {e.name}
                      {e.in_use ? <em className="engine-row__badge">已加载</em> : null}
                    </span>
                    <span className="engine-row__meta">
                      {e.size_mb.toFixed(0)} MB · {e.modified}
                    </span>
                    <span className="engine-row__dir">{e.dir}</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn--tiny btn--ghost"
                    title="在文件夹中显示"
                    onClick={() => props.onReveal?.(e.path)}
                  >
                    打开目录
                  </button>
                </li>
              );
            })}
          </ul>

          {incompatible.length > 0 ? (
            <details className="engine-panel__incompatible">
              <summary>{incompatible.length} 个非 laya 模型（其他项目的 engine）</summary>
              <ul>
                {incompatible.map((e: EngineFile) => (
                  <li key={e.path} title={`${e.path}\n${e.reason ?? ''}`}>
                    <span className="engine-row__name">{e.name}</span>
                    <span className="engine-row__reason">{e.reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </div>
      ) : (
        <p className="engine-panel__empty">
          当前模式下没有可用模型。请选择对应的 .engine / .onnx 文件。
        </p>
      )}

      {loaded ? (
        <>
          <dl className="engine-panel__summary">
            <div>
              <dt>并发</dt>
              <dd>{contextCount ?? '—'}</dd>
            </div>
            <div>
              <dt>激活显存</dt>
              <dd>{activationMB !== undefined ? `${activationMB.toFixed(1)} MiB` : '—'}</dd>
            </div>
          </dl>

          <details className="engine-panel__io">
            <summary>模型输入输出（{inputs.length} 进 / {outputs.length} 出）</summary>
            <div className="engine-panel__io-grid">
              <TensorTable caption="输入" tensors={inputs} />
              <TensorTable caption="输出" tensors={outputs} />
            </div>
          </details>
        </>
      ) : null}
    </section>
  );
};

const TensorTable: FC<{
  caption: string;
  tensors: Array<{ name: string; dtype: string; shape: number[] }>;
}> = ({ caption, tensors }) => (
  <table className="tensor-table">
    <caption>{caption}</caption>
    <thead>
      <tr>
        <th>名称</th>
        <th>类型</th>
        <th>形状</th>
      </tr>
    </thead>
    <tbody>
      {tensors.map((t) => (
        <tr key={t.name}>
          <td className="mono">{t.name}</td>
          <td className="mono">{formatDtype(t.dtype)}</td>
          <td className="mono">{formatShape(t.shape)}</td>
        </tr>
      ))}
      {tensors.length === 0 ? (
        <tr>
          <td colSpan={3} className="tensor-table__empty">
            无
          </td>
        </tr>
      ) : null}
    </tbody>
  </table>
);
