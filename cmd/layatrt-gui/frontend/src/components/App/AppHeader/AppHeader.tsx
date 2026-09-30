// AppHeader — view layer.

import type { FC } from 'react';
import type { AppHeaderProps } from './AppHeader.data';
import type { AppHeaderEvents } from './AppHeader.api';
import { useAppHeaderLogic } from './AppHeader';

export const AppHeader: FC<AppHeaderProps & AppHeaderEvents> = (props) => {
  const { vram, kernelLabel, deviceLabel, healthy, engineLabel } = useAppHeaderLogic(props);

  return (
    <header className="app-header">
      <div className="app-header__identity">
        <h1 className="app-header__title">{props.title}</h1>
        <span className="app-header__version">v{props.version}</span>
      </div>

      <dl className="app-header__facts">
        <div className="app-header__fact">
          <dt>模型</dt>
          <dd className={props.engineLoaded ? 'is-ok' : 'is-idle'}>{engineLabel}</dd>
        </div>
        <div className="app-header__fact">
          <dt>内核</dt>
          <dd className={healthy ? 'is-ok' : 'is-bad'}>{kernelLabel}</dd>
        </div>
        <div className="app-header__fact">
          <dt>显卡</dt>
          <dd>{deviceLabel}</dd>
        </div>
        <div className="app-header__fact">
          <dt>显存</dt>
          <dd>{vram}</dd>
        </div>
      </dl>

      {props.onRefresh ? (
        <button type="button" className="btn btn--ghost btn--tiny" onClick={props.onRefresh}>
          刷新
        </button>
      ) : null}
    </header>
  );
};
