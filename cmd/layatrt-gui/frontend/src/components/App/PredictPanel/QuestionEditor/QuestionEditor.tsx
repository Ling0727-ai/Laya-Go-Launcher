// QuestionEditor — view layer.
//
// Options are rows, one per criterion, not a textarea the user has to format.
// laya's `criteria` is a mapping (choice) or a list (score); rows map onto both
// mechanically, so there is no syntax to get wrong.
//
// The row's label is not an input: it is the position in the list, shown as a
// badge. A `choice` question reports its answer under that label, so deriving it
// is what keeps the labels unique and consecutive however rows are added or
// removed.

import type { FC } from 'react';
import type { QuestionEditorProps } from './QuestionEditor.data';
import { QUESTION_TYPE_OPTIONS } from './QuestionEditor.data';
import type { QuestionEditorEvents } from './QuestionEditor.api';
import { useQuestionEditorLogic } from './QuestionEditor';

export const QuestionEditor: FC<QuestionEditorProps & QuestionEditorEvents> = (props) => {
  const {
    draft,
    index,
    valid,
    options,
    filled,
    overCapacity,
    canRemoveOption,
    labelFor,
    update,
    setId,
    setType,
    setOption,
    addOption,
    removeOption,
    remove,
  } = useQuestionEditorLogic(props);

  const isScore = draft.type === 'score';
  const isNoUL = draft.type === 'noul';

  return (
    <fieldset className={`question-editor${valid ? '' : ' question-editor--invalid'}`}>
      <legend>
        <span className="question-editor__index">#{index + 1}</span>
        <input
          className="question-editor__id"
          type="text"
          value={draft.id}
          spellCheck={false}
          placeholder="问题 id"
          aria-label="问题 id"
          onChange={(event) => setId(event.target.value)}
        />
        <select
          className="question-editor__type"
          value={draft.type}
          aria-label="问题类型"
          onChange={(event) => setType(event.target.value as typeof draft.type)}
        >
          {QUESTION_TYPE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value} title={option.hint}>
              {option.label}
            </option>
          ))}
        </select>
        <button type="button" className="btn btn--danger btn--tiny" onClick={remove}>
          删除
        </button>
      </legend>

      <label className="field">
        <span className="field__label">问题</span>
        <input
          className="field__input"
          type="text"
          value={draft.instructions}
          placeholder="应该由哪个部门处理？"
          onChange={(event) => update({ instructions: event.target.value })}
        />
      </label>

      {isNoUL ? (
        <p className="question-editor__hint">
          <code>noul</code> 返回校准过的真假概率，不需要填选项。
        </p>
      ) : (
        <div className="option-rows">
          <div className="option-rows__head">
            <span>
              {isScore ? '等级（从上到下：低 → 高）' : '选项（标签自动生成，只填内容）'}
            </span>
            <span className="option-rows__count">
              {filled} / {options}
              {overCapacity ? ' · 超过模型上限' : ''}
            </span>
          </div>

          {draft.options.map((option, i) => (
            <div className="option-row" key={i}>
              <span
                className="option-row__badge"
                title={isScore ? `等级 ${i}（从上到下由低到高）` : `选项标签 ${labelFor(i)}`}
                aria-label={isScore ? `等级 ${i}` : `选项标签 ${labelFor(i)}`}
              >
                {labelFor(i)}
              </span>
              <input
                className="option-row__text"
                type="text"
                value={option.text}
                spellCheck={false}
                placeholder={isScore ? '这个等级的含义' : '选项内容（模型读这段）'}
                aria-label={isScore ? `等级 ${i} 含义` : `选项 ${labelFor(i)} 内容`}
                onChange={(event) => setOption(i, { text: event.target.value })}
              />
              <button
                type="button"
                className="btn btn--tiny btn--ghost"
                disabled={!canRemoveOption}
                title={canRemoveOption ? '删除这一项' : '至少保留两项'}
                aria-label={isScore ? `删除等级 ${i}` : `删除选项 ${labelFor(i)}`}
                onClick={() => removeOption(i)}
              >
                ✕
              </button>
            </div>
          ))}

          <button type="button" className="btn btn--tiny option-rows__add" onClick={addOption}>
            + 添加选项
          </button>
        </div>
      )}
    </fieldset>
  );
};
