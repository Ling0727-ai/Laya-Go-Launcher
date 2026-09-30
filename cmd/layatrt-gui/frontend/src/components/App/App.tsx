// App — view layer. Mounts the panels; holds no business logic.

import type { FC } from 'react';
import type { AppProps } from './App.data';
import { useAppLogic } from './App';
import { AppHeader } from './AppHeader/AppHeader.tsx';
import { EnginePanel } from './EnginePanel/EnginePanel.tsx';
import { LauncherPanel } from './LauncherPanel/LauncherPanel.tsx';
import { MetricsPanel } from './MetricsPanel/MetricsPanel.tsx';
import { PredictPanel } from './PredictPanel/PredictPanel.tsx';

export const App: FC<AppProps> = (props) => {
  const app = useAppLogic(props);

  if (!app.bindingsReady) {
    return (
      <div className="app app--standalone">
        <div className="app__notice">
          <h1>Laya Go Launcher</h1>
          <p>这个页面没有运行在桌面窗口里，所以拿不到本机能力。</p>
          <p>
            用 <code>.\dev.ps1</code> 启动桌面端，或直接访问 HTTP API：
            <code>http://127.0.0.1:8420/api/v1</code>
          </p>
        </div>
      </div>
    );
  }

  const device = app.status?.device;
  const kernel = app.status?.kernel;
  const engineLoaded = app.status?.engine.loaded ?? false;

  return (
    <div className="app">
      <AppHeader
        title="Laya Go Launcher"
        version={app.version}
        kernelAvailable={kernel?.available ?? false}
        tensorrtVersion={kernel?.tensorrt}
        deviceName={device?.name ?? ''}
        computeCapability={device?.compute_capability ?? '0.0'}
        vramFreeMB={device?.vram_free_mb ?? 0}
        vramTotalMB={device?.vram_total_mb ?? 0}
        engineLoaded={engineLoaded}
        onRefresh={() => void app.refresh()}
      />

      <main className="app__body">
        <div className="app__column app__column--left">
          <LauncherPanel onModelChanged={() => void app.refresh()} />
          <EnginePanel
            mode={app.engineMode}
            engine={app.status?.engine ?? { loaded: false }}
            discovered={app.engines}
            busy={app.loadingEngine || app.predicting}
            scanning={app.scanning}
            error={app.engineError}
            onLoad={(selection) => void app.handleLoadEngine(selection)}
            onModeChange={app.handleModeChange}
            onUnload={() => void app.handleUnload()}
            onBrowse={() => void app.handleBrowseEngine()}
            onRescan={() => void app.rescan()}
            onReveal={(path) => void app.handleReveal(path)}
          />
          <MetricsPanel metrics={app.metrics} onRefresh={() => void app.refresh()} />
        </div>

        <div className="app__column app__column--right">
          <PredictPanel
            engineLoaded={engineLoaded}
            limits={app.limits}
            result={app.result}
            busy={app.predicting}
            loadingFile={app.loadingFile}
            error={app.predictError}
            stateText={app.stateText}
            stateSource={app.stateSource}
            questions={app.questions}
            questionLegend={app.resultLegend}
            onStateChange={app.setStateText}
            onQuestionsChange={app.setQuestions}
            onLoadDocument={() => void app.handleLoadDocument()}
            onSubmit={(state, questions, legend) =>
              void app.handlePredict(state, questions, legend)
            }
            onClear={app.handleClear}
          />
        </div>
      </main>
    </div>
  );
};
