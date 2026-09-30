// PredictPanel — view layer.
//
// This is a decision model: the questions are the point. The editor is the main
// workspace, not a collapsed detail, and a template is only a starting point that
// fills it (the user is expected to change what it fills in).
//
// A result names the chosen option by its label *and* its content: the engine
// answers with the label (A, B, … for choice, the level index for score), which
// on its own does not say what was chosen.

import type { FC } from 'react';
import type { Answer } from '@/types/api';
import { formatMs, formatProbability, formatPercent } from '@/utils/format';
import type { QuestionLegend, PredictPanelProps } from './PredictPanel.data';
import type { PredictPanelEvents } from './PredictPanel.api';
import { usePredictPanelLogic } from './PredictPanel';
import { QuestionEditor } from './QuestionEditor/QuestionEditor.tsx';

export const PredictPanel: FC<PredictPanelProps & PredictPanelEvents> = (props) => {
  const {
    questions,
    validCount,
    invalidCount,
    overCapacity,
    capacity,
    submit,
    addQuestion,
    removeQuestion,
    updateQuestion,
    canSubmit,
    answers,
    timing,
    usage,
    model,
  } = usePredictPanelLogic(props);

  return (
    <section className="panel predict-panel">
      <h2 className="panel__title">决策问题</h2>

      <div className="questions-list">
        {props.questions.map((draft, index) => (
          <QuestionEditor
            key={`${index}-${draft.id}`}
            draft={draft}
            index={index}
            valid={Boolean(questions[draft.id.trim()])}
            markerCapacity={capacity}
            onChange={(next) => updateQuestion(index, next)}
            onRemove={() => removeQuestion(index)}
          />
        ))}
      </div>

      <div className="predict-panel__question-actions">
        <button type="button" className="btn btn--tiny" onClick={addQuestion}>
          + 添加问题
        </button>
        {overCapacity.length > 0 ? (
          <span className="predict-panel__hint predict-panel__hint--warn">
            {overCapacity.length} 个问题的选项数超过模型上限 {capacity}
          </span>
        ) : null}
        {invalidCount > 0 ? (
          <span className="predict-panel__hint">{invalidCount} 个未填写完整</span>
        ) : null}
      </div>

      <hr className="predict-panel__divider" />

      <div className="state-bar">
        <button
          type="button"
          className="btn"
          disabled={props.loadingFile}
          onClick={() => props.onLoadDocument?.()}
        >
          {props.loadingFile ? '读取中…' : '打开文件…'}
        </button>
        {props.stateSource ? (
          <span className="state-bar__source mono" title={props.stateSource}>
            {props.stateSource}
          </span>
        ) : (
          <span className="state-bar__count">{props.stateText.length} 字符</span>
        )}
      </div>

      <label className="field">
        <span className="field__label">待分析内容</span>
        <textarea
          className="field__input field__input--area"
          rows={7}
          value={props.stateText}
          spellCheck={false}
          placeholder="粘贴一段文本，或用「打开文件…」选一个文件"
          onChange={(event) => props.onStateChange?.(event.target.value)}
        />
      </label>

      <div className="predict-panel__actions">
        <button
          type="button"
          className="btn btn--primary btn--run"
          disabled={!canSubmit}
          onClick={submit}
        >
          {props.busy ? '推理中…' : `运行（${validCount} 个问题）`}
        </button>
        <button type="button" className="btn btn--ghost" onClick={() => props.onClear?.()}>
          清空结果
        </button>
      </div>

      {!props.engineLoaded ? (
        <p className="predict-panel__notice">先在左侧加载一个模型。</p>
      ) : null}
      {props.error ? <p className="predict-panel__error">{props.error}</p> : null}

      {props.result ? (
        <div className="predict-panel__result">
          <div className="predict-panel__result-head">
            <h3>{model}</h3>
            <span>
              {formatMs(timing?.total_ms)} 总计 · {formatMs(timing?.inference_ms)} 推理 ·{' '}
              {usage?.input_tokens ?? 0} tokens
            </span>
          </div>
          <div className="answer-list">
            {Object.entries(answers).map(([id, answer]) => (
              <AnswerCard
                key={id}
                id={id}
                answer={answer}
                legend={props.questionLegend?.[id]}
              />
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
};

/** The content behind a reported label, or '' when it is not known. */
function contentFor(
  legend: Record<string, string> | undefined,
  label: string,
): string {
  const text = legend?.[label];
  // A choice question's criteria fall back to the content as its own label, so
  // suppress a content that just repeats the label.
  return text && text !== label ? text : '';
}

const AnswerCard: FC<{
  id: string;
  answer: Answer;
  legend?: Record<string, string>;
}> = ({ id, answer, legend }) => {
  // The panel's snapshot is preferred: it is taken when the request is built, so
  // it survives the editor changing afterwards. A score question also carries a
  // legend of its own.
  const labels = legend ?? answer.legend;
  const chosen = answer.choice ?? '';
  const chosenText = contentFor(labels, chosen);

  return (
    <article className="answer-card">
      <header className="answer-card__head">
        <span className="answer-card__id mono">{id}</span>
        <span className="answer-card__type">{answer.type}</span>
        <span className="answer-card__confidence" title="归一化熵置信度">
          置信 {formatProbability(answer.confidence)}
        </span>
      </header>

      {answer.type === 'choice' ? (
        <>
          <p className="answer-card__value">
            <span className="option-badge">{chosen || '—'}</span>
            {chosenText ? <span className="answer-card__value-text">{chosenText}</span> : null}
          </p>
          <ProbabilityBars probabilities={answer.probabilities ?? {}} legend={labels} />
        </>
      ) : null}

      {answer.type === 'score' ? (
        <>
          <p className="answer-card__value">{answer.score?.toFixed(3) ?? '—'}</p>
          <ProbabilityBars probabilities={answer.probabilities ?? {}} legend={labels} />
        </>
      ) : null}

      {answer.type === 'noul' ? (
        <p className="answer-card__value">
          P(true) {formatProbability(answer.noul)} <small>({formatPercent(answer.noul)})</small>
        </p>
      ) : null}

      <footer className="answer-card__foot">
        动作 {formatProbability(answer.action?.act_probability)}
      </footer>
    </article>
  );
};

const ProbabilityBars: FC<{
  probabilities: Record<string, number>;
  legend?: Record<string, string>;
}> = ({ probabilities, legend }) => {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <ul className="prob-bars">
      {entries.map(([label, value]) => {
        const text = contentFor(legend, label);
        return (
          <li key={label} className="prob-bars__row">
            <span className="option-badge option-badge--small" title={label}>
              {label}
            </span>
            <span className="prob-bars__label" title={text || label}>
              {text || label}
            </span>
            <span className="prob-bars__track">
              <span className="prob-bars__fill" style={{ width: `${Math.max(0, value) * 100}%` }} />
            </span>
            <span className="prob-bars__value mono">{formatProbability(value)}</span>
          </li>
        );
      })}
    </ul>
  );
};
