// Package sequence builds the token layout the laya decision head expects.
//
// It is a faithful port of `build_sequence` in the Python SDK's laya/common.py:
//
//	[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//
// The head does not read a pooled sentence vector; it scores the hidden state at
// each [MASK] position, one per option. So the marker positions are as much part
// of the contract as the ids are, and getting the head budget wrong silently
// truncates options into each other.
package sequence

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/local/laya-go-launcher/internal/tokenizer"
)

// QType is the decision primitive. The values are the head's type embedding
// indices and must not change.
type QType int

const (
	Choice QType = 0
	Score  QType = 1
	NoUL   QType = 2
)

// String names the primitive as it appears in API payloads.
func (q QType) String() string {
	switch q {
	case Choice:
		return "choice"
	case Score:
		return "score"
	case NoUL:
		return "noul"
	default:
		return fmt.Sprintf("qtype(%d)", int(q))
	}
}

// ParseQType maps an API string onto a primitive.
func ParseQType(s string) (QType, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "choice":
		return Choice, nil
	case "score":
		return Score, nil
	case "noul":
		return NoUL, nil
	default:
		return 0, fmt.Errorf("sequence: unknown question type %q (want choice, score or noul)", s)
	}
}

// Question is one typed question, already reduced to the form the builder needs.
type Question struct {
	Type         QType
	Instructions string
	// Criteria holds rendered option texts in label order. For noul it is
	// always the two synthetic options [false, true].
	Criteria []string
	// Labels parallels Criteria for choice questions, in the same order, so the
	// answer can be mapped back from an index to a label.
	Labels []string
}

// Built is a fully tokenised question sequence.
type Built struct {
	IDs             []int
	MarkerPositions []int
	Options         []string
	QType           QType
	Truncated       bool
}

// Options renders a criterion value the way the Python SDK does: strings pass
// through, anything structured becomes compact JSON.
func renderCriterion(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(raw)
}

// RenderOptions produces the option strings for a question, in label order.
func RenderOptions(q Question) []string {
	if q.Type != NoUL {
		return q.Criteria
	}
	return []string{
		"false: no, the statement does not hold",
		"true: yes, the statement holds",
	}
}

// maskToken is the literal the checkpoints use at the positions the head
// scores. Text that is not meant to carry a marker has it neutralised to a
// space, exactly as the SDK does.
const maskToken = "[MASK]"

// EncodeState tokenises the state once so a batch of questions over the same
// state does not re-encode it per question.
//
// This is the expensive half of Build for a long document: the state is the
// only part of the sequence that does not depend on the question. Encoding it
// inside a per-question loop made tokenisation dominate total latency on long
// inputs, and the result was then truncated to a fraction of its length anyway.
//
// budget caps how many tokens are produced. Pass the request's maxLen: no
// question can place more state tokens than that, and the layout keeps the head
// of the state, so encoding further is work that is always discarded. A budget
// of 0 or less means "no limit" and encodes the whole state.
//
// The marker literal is neutralised here rather than in the caller because a
// state is never a place where [MASK] should be honoured.
func EncodeState(tk *tokenizer.Tokenizer, state string, budget int) []int {
	if tk == nil {
		return nil
	}
	text := strings.ReplaceAll(state, maskToken, " ")
	if budget <= 0 {
		return tk.Encode(text)
	}
	return tk.EncodePrefix(text, budget)
}

// Build lays out the sequence for one question, tokenising the state itself.
//
// Prefer BuildWithState when several questions share one state; this wrapper
// pays for a full state encoding on every call.
func Build(tk *tokenizer.Tokenizer, state string, q Question, maxLen, headMaxLen int) (Built, error) {
	return BuildWithState(tk, EncodeState(tk, state, maxLen), q, maxLen, headMaxLen)
}

// BuildWithState is Build with the state already tokenised by EncodeState.
//
// maxLen is the total token budget and headMaxLen the budget for the question
// header plus its options; whatever remains holds the state. The Python SDK
// defaults are 512/192 for the English checkpoint.
//
// headMaxLen is a soft budget: when the natural option lengths overflow it,
// every option is compressed to an equal share (never below 4 tokens) rather
// than dropped. Options are only lost if maxLen truncates past their markers,
// and that is reported as an error because the head would then score fewer
// options than the question declares.
//
// stateIDs is read-only and shared across a batch, so it is sliced to the
// available room and never written to.
func BuildWithState(tk *tokenizer.Tokenizer, stateIDs []int, q Question, maxLen, headMaxLen int) (Built, error) {
	if tk == nil {
		return Built{}, fmt.Errorf("sequence: tokenizer is required")
	}
	if maxLen <= 0 {
		return Built{}, fmt.Errorf("sequence: maxLen must be positive")
	}
	if headMaxLen <= 0 {
		return Built{}, fmt.Errorf("sequence: headMaxLen must be positive")
	}

	opts := RenderOptions(q)

	// The instructions are tokenised without specials and with the mask token
	// neutralised, matching the SDK.
	ins := strings.ReplaceAll(q.Instructions, maskToken, " ")
	headText := fmt.Sprintf("%s question: %s", q.Type, ins)
	headIDs := tk.Encode(headText)

	// Each option is a leading [MASK] plus its text, capped at 48 tokens.
	optIDs := make([][]int, len(opts))
	for i, opt := range opts {
		body := tk.Encode(" " + strings.ReplaceAll(opt, maskToken, " "))
		if len(body) > 48 {
			body = body[:48]
		}
		optIDs[i] = append([]int{tk.MaskID()}, body...)
	}

	// Compress options until the header plus options fit the head budget,
	// keeping at least 16 tokens for the instruction text.
	optBudget := headMaxLen - totalLen(optIDs)
	if optBudget < 16 {
		per := (headMaxLen - 16) / max(1, len(optIDs))
		if per < 4 {
			per = 4
		}
		for i := range optIDs {
			if len(optIDs[i]) > per {
				optIDs[i] = optIDs[i][:per]
			}
		}
		optBudget = headMaxLen - totalLen(optIDs)
	}

	headKeep := optBudget
	if headKeep < 8 {
		headKeep = 8
	}
	if headKeep > len(headIDs) {
		headKeep = len(headIDs)
	}
	headIDs = headIDs[:headKeep]

	ids := make([]int, 0, maxLen)
	ids = append(ids, tk.CLSID())
	ids = append(ids, headIDs...)
	ids = append(ids, tk.SEPID())

	markers := make([]int, 0, len(optIDs))
	for _, o := range optIDs {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, tk.SEPID())

	// Whatever budget is left goes to the state, truncated from the right.
	//
	// stateIDs is shared across a batch, so it is sliced rather than re-encoded
	// or mutated. Truncation from the right keeps the head of the document,
	// which is where the SDK puts the useful context.
	room := maxLen - len(ids) - 1
	if room < 0 {
		room = 0
	}
	truncated := false
	if len(stateIDs) > room {
		stateIDs = stateIDs[:room]
		truncated = true
	}
	ids = append(ids, stateIDs...)
	ids = append(ids, tk.SEPID())

	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}

	// Markers beyond the truncation point are dropped, as the SDK does.
	kept := markers[:0]
	for _, m := range markers {
		if m < maxLen {
			kept = append(kept, m)
		}
	}

	built := Built{
		IDs:             ids,
		MarkerPositions: kept,
		Options:         opts,
		QType:           q.Type,
		Truncated:       truncated,
	}
	if len(built.MarkerPositions) != len(opts) {
		return built, fmt.Errorf(
			"sequence: %d of %d option markers survived max_len=%d; "+
				"raise max_len or reduce the option count",
			len(built.MarkerPositions), len(opts), maxLen)
	}
	return built, nil
}

// PadTo returns the ids padded with padID to exactly length n, plus a mask that
// is 1 for real tokens and 0 for padding. When ids is already longer it is
// truncated.
//
// Padding is safe: the encoder honours the attention mask, so a padded run
// returns the same decision as the unpadded one.
//
// Prefer PadInto when the destination already exists; this allocates two slices
// per call.
func PadTo(ids []int, n, padID int) ([]int64, []int64) {
	out := make([]int64, n)
	mask := make([]int64, n)
	PadInto(ids, out, mask, padID)
	return out, mask
}

// PadInto writes the padded ids and attention mask into caller-owned slices,
// which must both be at least as long as the requested length. It returns the
// number of positions written.
//
// This exists for the batch path, where the destination is one row of a
// pre-allocated batch array: calling PadTo there allocated two slices per row
// only to copy them into place and discard them.
func PadInto(ids []int, dst, mask []int64, padID int) int {
	n := len(dst)
	if len(mask) < n {
		n = len(mask)
	}
	if n <= 0 {
		return 0
	}
	real := len(ids)
	if real > n {
		real = n
	}
	for i := 0; i < real; i++ {
		dst[i] = int64(ids[i])
		mask[i] = 1
	}
	for i := real; i < n; i++ {
		dst[i] = int64(padID)
		mask[i] = 0
	}
	return n
}

func totalLen(v [][]int) int {
	n := 0
	for _, x := range v {
		n += len(x)
	}
	return n
}
