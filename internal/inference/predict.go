// Package inference is the predict use case: turn a state and a set of typed
// questions into calibrated answers, in one forward pass.
//
// This is where the three lower layers meet — the tokenizer, the sequence
// builder, and the execution kernel — and it is the only place that knows the
// model's IO contract (five int64/bool inputs, two float32 outputs).
//
// The kernel is reached through backend.Backend, so this package names no
// TensorRT or ONNX Runtime type. Which kernel serves a request is decided in
// internal/backends and is invisible here.
package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/sequence"
	"github.com/local/laya-go-launcher/internal/tokenizer"
)

// QuestionSpec is a question as the caller writes it.
//
// The JSON tags are the wire format documented in docs/api.md and read by the
// GUI, so they have to stay in step with those.
type QuestionSpec struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	// Criteria is either a map of label -> description (choice) or a list of
	// level descriptions (score). It is nil for noul.
	Criteria any `json:"criteria,omitempty"`
}

// Request is one predict call.
type Request struct {
	State     any
	Questions map[string]QuestionSpec
}

// Action carries the head's auxiliary action logits.
type Action struct {
	ActProbability float64 `json:"act_probability"`
}

// Answer is one question's result. Exactly one of Choice, Score or NoUL is set,
// according to the question type.
type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence"`
	Action        Action             `json:"action"`

	Score   float64           `json:"score,omitempty"`
	Legend  map[string]string `json:"legend,omitempty"`
	NoUL    float64           `json:"noul,omitempty"`
	NoULSet bool              `json:"-"`
}

// Usage reports token accounting.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Timing breaks the call down.
type Timing struct {
	TotalMS     float64 `json:"total_ms"`
	TokenizeMS  float64 `json:"tokenize_ms"`
	InferenceMS float64 `json:"inference_ms"`
}

// Response is a completed predict.
type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
	Timing  Timing            `json:"timing"`
	// Routing is reserved for a future multi-checkpoint router; the English
	// checkpoint is the only one wired up today.
	Routing map[string]any `json:"routing,omitempty"`
}

// Service holds the model and the two knobs that control the head budget.
type Service struct {
	model backend.Backend
	tok   *tokenizer.Tokenizer

	// maxLen and headMaxLen are the token budgets, derived from the model by
	// AdaptToEngine. padLen is the fixed sequence length a static model demands,
	// or 0 for a dynamic one.
	mu         sync.RWMutex
	maxLen     int
	headMaxLen int
	padLen     int

	// modelCfg is the checkpoint's rl_agent_config.json. Its calibration
	// temperatures are applied to every logit row, so this is not optional for
	// correct probabilities.
	modelCfgMu sync.RWMutex
	modelCfg   *ModelConfig
}

// Config configures a Service.
type Config struct {
	// Backend is the execution kernel. Required.
	Backend backend.Backend
	// Tokenizer is the model's tokenizer. Required.
	Tokenizer *tokenizer.Tokenizer
	// ModelConfig is the checkpoint's rl_agent_config.json. When nil the service
	// runs with no temperature scaling, which is only correct for a checkpoint
	// whose temperatures are all 1.0.
	ModelConfig *ModelConfig
	MaxLen      int
	HeadMaxLen  int
	PadLen      int
}

// SetModelConfig installs the checkpoint configuration.
func (s *Service) SetModelConfig(cfg *ModelConfig) {
	s.modelCfgMu.Lock()
	s.modelCfg = cfg
	s.modelCfgMu.Unlock()
}

// ModelConfig returns the installed checkpoint configuration, or nil.
func (s *Service) ModelConfig() *ModelConfig {
	s.modelCfgMu.RLock()
	defer s.modelCfgMu.RUnlock()
	return s.modelCfg
}

// LoadModelConfigFrom loads the checkpoint configuration out of a directory (or
// a config file path), installs it, and re-fits the token budgets.
//
// This is the step that makes probabilities match the trained calibration; a
// service without it silently uses temperature 1.0 for every question.
func (s *Service) LoadModelConfigFrom(path string) error {
	cfg, err := LoadModelConfig(path)
	if err != nil {
		return err
	}
	s.SetModelConfig(cfg)
	s.AdaptToEngine()
	return nil
}

// NewService wires a service to a backend and tokenizer.
//
// MaxLen/HeadMaxLen/PadLen of 0 mean "ask the model": call AdaptToEngine after
// loading a model and the budgets come from its input shapes. Non-zero values
// are treated as a starting point that AdaptToEngine will still correct for a
// fixed-shape model, because such a model accepts exactly one length.
func NewService(cfg Config) (*Service, error) {
	if cfg.Backend == nil {
		return nil, fmt.Errorf("inference: backend is required")
	}
	if cfg.Tokenizer == nil {
		return nil, fmt.Errorf("inference: tokenizer is required")
	}
	s := &Service{
		model:      cfg.Backend,
		tok:        cfg.Tokenizer,
		maxLen:     cfg.MaxLen,
		headMaxLen: cfg.HeadMaxLen,
		padLen:     cfg.PadLen,
		modelCfg:   cfg.ModelConfig,
	}
	// Fall back to the checkpoint's own defaults only when nothing else is known.
	// These are the values in rl_agent_config.json for the English checkpoint.
	if s.maxLen <= 0 {
		s.maxLen = 512
	}
	if s.headMaxLen <= 0 {
		s.headMaxLen = 192
	}
	return s, nil
}

// Budgets returns the current token budgets.
func (s *Service) Budgets() (maxLen, headMaxLen, padLen int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxLen, s.headMaxLen, s.padLen
}

// SetBudgets changes the token budgets. padLen is the fixed sequence length the
// engine expects, or 0 for a dynamic-shape engine.
func (s *Service) SetBudgets(maxLen, headMaxLen, padLen int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if maxLen > 0 {
		s.maxLen = maxLen
	}
	if headMaxLen > 0 {
		s.headMaxLen = headMaxLen
	}
	s.padLen = padLen
}

// AdaptToEngine fits the token budgets to the loaded engine and the checkpoint.
//
// Three numbers are in play and they mean different things:
//
//   - The engine's optimisation profile bounds the *possible* range. A plan built
//     with maxShapes=input_ids:1x8192 accepts up to 8192 tokens.
//   - The checkpoint's rl_agent_config.json records what the model was *trained*
//     at (512 total, 192 for the question head on the English checkpoint).
//   - The engine's build shape, for a fixed plan, is the only length accepted.
//
// The default is the trained value, clamped into the engine's range: running
// beyond the training context is supported but is a deliberate choice, not
// something a first run should do silently. Raise MaxLen/HeadMaxLen explicitly
// (or via the config file) to use more of the engine.
//
// Returns the budgets it settled on.
func (s *Service) AdaptToEngine() (maxLen, headMaxLen, padLen int) {
	seqMin, seqMax, seqOK := s.model.InputBounds("input_ids", 1)
	_, markerMax, markerOK := s.model.InputBounds("marker_pos", 1)

	s.modelCfgMu.RLock()
	trainedMax, trainedHead := 0, 0
	if s.modelCfg != nil {
		trainedMax, trainedHead = s.modelCfg.MaxLen, s.modelCfg.HeadMaxLen
	}
	s.modelCfgMu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()

	if !seqOK || seqMax <= 0 {
		// Nothing to learn from the engine; keep whatever is configured.
		return s.maxLen, s.headMaxLen, s.padLen
	}

	// Start from the trained values, or the current ones when no config is loaded.
	wantMax := trainedMax
	if wantMax <= 0 {
		wantMax = s.maxLen
	}
	wantHead := trainedHead
	if wantHead <= 0 {
		wantHead = s.headMaxLen
	}

	// Clamp into what the engine accepts.
	if wantMax > seqMax {
		wantMax = seqMax
	}
	if seqMin > 0 && wantMax < seqMin {
		wantMax = seqMin
	}
	if wantHead > wantMax {
		wantHead = wantMax
	}

	// Every option needs a [MASK] plus a token or two; never budget less than
	// that or options are squeezed to nothing.
	if markerOK && markerMax > 0 {
		if minHead := markerMax * 4; wantHead < minHead {
			wantHead = minHead
		}
	}
	if wantHead < 16 {
		wantHead = 16
	}
	if wantHead > wantMax {
		wantHead = wantMax
	}

	s.maxLen = wantMax
	s.headMaxLen = wantHead
	if seqMin == seqMax {
		// Fixed-shape plan: every request is padded to this exact length.
		s.padLen = seqMax
	} else {
		// Dynamic: let the sequence take its natural length inside the range.
		s.padLen = 0
	}
	return s.maxLen, s.headMaxLen, s.padLen
}

// Limits describes what the loaded engine accepts, what the checkpoint was
// trained at, and what the service is using.
type Limits struct {
	// SequenceMin/Max are the token counts the engine accepts.
	SequenceMin int `json:"sequence_min"`
	SequenceMax int `json:"sequence_max"`
	// MarkersMax is the most options one question may declare.
	MarkersMax int `json:"markers_max"`
	// FixedLength is true when the engine accepts exactly one sequence length.
	FixedLength bool `json:"fixed_length"`
	// TrainedMaxLen/TrainedHeadMaxLen come from the checkpoint config, or 0 when
	// no config is loaded.
	TrainedMaxLen     int `json:"trained_max_len"`
	TrainedHeadMaxLen int `json:"trained_head_max_len"`
	// Budgets are the values currently in effect.
	MaxLen     int `json:"max_len"`
	HeadMaxLen int `json:"head_max_len"`
	PadLen     int `json:"pad_len"`
}

// Limits reports what the engine accepts and what the service is using.
func (s *Service) Limits() Limits {
	seqMin, seqMax, seqOK := s.model.InputBounds("input_ids", 1)
	_, markerMax, markerOK := s.model.InputBounds("marker_pos", 1)

	s.modelCfgMu.RLock()
	var trainedMax, trainedHead int
	if s.modelCfg != nil {
		trainedMax, trainedHead = s.modelCfg.MaxLen, s.modelCfg.HeadMaxLen
	}
	s.modelCfgMu.RUnlock()

	s.mu.RLock()
	defer s.mu.RUnlock()

	l := Limits{
		SequenceMin:       seqMin,
		SequenceMax:       seqMax,
		TrainedMaxLen:     trainedMax,
		TrainedHeadMaxLen: trainedHead,
		MaxLen:            s.maxLen,
		HeadMaxLen:        s.headMaxLen,
		PadLen:            s.padLen,
	}
	if markerOK {
		l.MarkersMax = markerMax
	}
	l.FixedLength = seqOK && seqMin == seqMax
	return l
}

// MarkerCapacity is how many options the engine can score in one pass, or 0 when
// unknown.
func (s *Service) MarkerCapacity() int {
	_, markerMax, ok := s.model.InputBounds("marker_pos", 1)
	if !ok {
		return 0
	}
	return markerMax
}

// Tokenizer exposes the tokenizer for the inspect endpoints.
func (s *Service) Tokenizer() *tokenizer.Tokenizer { return s.tok }

// Backend exposes the execution kernel, for the inspect and diagnostics
// endpoints. The concrete type is deliberately not named, so this package stays
// independent of which kernel is installed.
func (s *Service) Backend() backend.Backend { return s.model }

// modelFixedShape returns the model's declared shape for a named input.
//
// A model built with a fixed dimension only accepts that exact size, so the
// request has to be padded to it: sequence length and marker count both have to
// match. Dynamic dimensions come back as -1 and impose nothing.
func (s *Service) modelFixedShape(name string) []int {
	info, ok := s.model.Info()
	if !ok {
		return nil
	}
	for _, t := range info.Inputs {
		if t.Name == name {
			return t.Shape
		}
	}
	return nil
}

// maxBatch returns how many sequence rows the loaded engine accepts at once.
//
// A dynamic batch dimension reports -1 in the engine's shape, so its range has
// to come from the optimisation profile. Reading only the static shape treated
// every dynamic-batch plan as batch-1, which silently disabled batching for the
// most common kind of plan: Predict then ran one forward pass per question and
// paid the per-run fixed cost each time.
//
// The result is capped at maxBatchCap so one request cannot be padded into a
// batch so large that the padding rows cost more than the saved forward passes.
func (s *Service) maxBatch() int {
	shape := s.modelFixedShape("input_ids")
	profMax, profOK := 0, false
	if _, max, ok := s.model.InputBounds("input_ids", 0); ok {
		profMax, profOK = max, true
	}
	return chooseMaxBatch(shape, profMax, profOK)
}

// chooseMaxBatch decides the batch size from the engine's declared shape and
// the profile's upper bound. It is a pure function so the decision can be
// tested without loading a plan.
func chooseMaxBatch(shape []int, profMax int, profOK bool) int {
	// A fixed batch dimension above 1 is the exact size the engine accepts; the
	// caller pads up to it.
	if len(shape) > 0 && shape[0] > 1 {
		return shape[0]
	}
	// Otherwise the batch dimension is 1 or dynamic. A dynamic plan declares its
	// range in the optimisation profile; a plan fixed at one row has no usable
	// range and must stay single-row.
	if !profOK {
		// A symbolic batch dimension with no profile (an ONNX graph) accepts any
		// batch, so batch up to the cap. Without this every question paid its own
		// forward pass on the ONNX kernel.
		if len(shape) > 0 && shape[0] < 0 {
			return maxBatchCap
		}
		return 1
	}
	if profMax < 2 {
		return 1
	}
	if profMax > maxBatchCap {
		return maxBatchCap
	}
	return profMax
}

// maxBatchCap bounds how many questions one forward pass may carry.
const maxBatchCap = 16

// fixedDim returns the engine's fixed value for one dimension, or 0 when the
// dimension is dynamic.
func fixedDim(shape []int, idx int) int {
	if idx < len(shape) && shape[idx] > 0 {
		return shape[idx]
	}
	return 0
}

// Predict runs every question, batching as many as the engine allows.
func (s *Service) Predict(ctx context.Context, req Request) (Response, error) {
	start := time.Now()
	if len(req.Questions) == 0 {
		return Response{}, fmt.Errorf("inference: at least one question is required")
	}

	state := SerializeState(req.State)

	// Question order must be stable: map iteration is random, and the answers
	// have to line up with the batch rows.
	ids := make([]string, 0, len(req.Questions))
	for id := range req.Questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	type prepared struct {
		id    string
		spec  QuestionSpec
		q     sequence.Question
		built sequence.Built
	}
	items := make([]preparedQuestion, 0, len(ids))

	s.mu.RLock()
	maxLen, headMaxLen, padLen := s.maxLen, s.headMaxLen, s.padLen
	s.mu.RUnlock()

	tokenizeStart := time.Now()
	// The state is the same for every question in a request and is the only
	// expensive part of the layout on a long document, so it is tokenised once
	// here rather than inside the per-question loop.
	//
	// Bounded by maxLen: no question can use more state tokens than that, and
	// the layout keeps the head of the state, so encoding past the budget is
	// work whose result is always discarded.
	stateIDs := sequence.EncodeState(s.tok, state, maxLen)
	for _, id := range ids {
		spec := req.Questions[id]
		q, err := BuildQuestion(spec)
		if err != nil {
			return Response{}, fmt.Errorf("question %q: %w", id, err)
		}
		built, err := sequence.BuildWithState(s.tok, stateIDs, q, maxLen, headMaxLen)
		if err != nil {
			return Response{}, fmt.Errorf("question %q: %w", id, err)
		}
		items = append(items, preparedQuestion{id: id, spec: spec, q: q, built: built})
	}
	tokenizeMS := msSince(tokenizeStart)

	batchLimit := s.maxBatch()
	inferStart := time.Now()
	answers := make(map[string]Answer, len(items))
	totalTokens := 0

	for start := 0; start < len(items); start += batchLimit {
		end := start + batchLimit
		if end > len(items) {
			end = len(items)
		}
		chunk := items[start:end]

		logitsRows, actRows, err := s.runChunkSplitting(ctx, chunk, padLen)
		if err != nil {
			return Response{}, err
		}

		for row, it := range chunk {
			if row >= len(logitsRows) {
				return Response{}, fmt.Errorf("inference: missing logits row %d", row)
			}
			k := len(it.built.MarkerPositions)
			raw := logitsRows[row]
			if k > len(raw) {
				k = len(raw)
			}
			// Apply the checkpoint's calibration temperature before the softmax.
			// The SDK does the same (t_scale = temperature_by_options[...] then
			// logits / t_scale); skipping it gives probabilities that do not match
			// the trained calibration, so confidence gating would be wrong.
			scale := s.modelCfg.TemperatureFor(it.q.Type.String(), k)
			probs, err := softmaxWithTemperature(raw[:k], scale)
			if err != nil {
				return Response{}, fmt.Errorf("question %q: %w", it.id, err)
			}
			actProb := 0.0
			if row < len(actRows) && len(actRows[row]) > 0 {
				actProb = actProbability(actRows[row])
			}
			answers[it.id] = buildAnswer(it.spec, it.q, it.built, probs, actProb)
			totalTokens += len(it.built.IDs)
		}
	}
	inferenceMS := msSince(inferStart)

	return Response{
		Model:   "laya-rl-agent",
		Answers: answers,
		Usage:   Usage{InputTokens: totalTokens, OutputTokens: 0},
		Timing: Timing{
			TotalMS:     msSince(start),
			TokenizeMS:  tokenizeMS,
			InferenceMS: inferenceMS,
		},
	}, nil
}

// debugInference enables per-run tracing of the engine's outputs.
var debugInference = os.Getenv("LAYA_TRT_DEBUG") != ""

// preparedQuestion is one question after tokenisation, ready to batch.
type preparedQuestion struct {
	id    string
	spec  QuestionSpec
	q     sequence.Question
	built sequence.Built
}

// runChunkSplitting runs a chunk, halving it when the engine has no profile for
// that batch at that length.
//
// A multi-profile plan reports the union of its profiles as its bounds (batch up
// to 8, length up to 8192), but no single profile admits every combination: the
// batched profile stops at the trained length and the long profile is batch 1.
// The native side rejects such a shape before any GPU work, so retrying with
// smaller batches costs only the rejected call.
func (s *Service) runChunkSplitting(ctx context.Context, items []preparedQuestion, padLen int) ([][]float32, [][]float32, error) {
	logits, acts, err := s.runChunk(ctx, items, padLen)
	if err == nil || len(items) < 2 || !isNoProfileError(err) {
		return logits, acts, err
	}
	mid := len(items) / 2
	l1, a1, err := s.runChunkSplitting(ctx, items[:mid], padLen)
	if err != nil {
		return nil, nil, err
	}
	l2, a2, err := s.runChunkSplitting(ctx, items[mid:], padLen)
	if err != nil {
		return nil, nil, err
	}
	// runChunk may pad a fixed-batch plan with extra rows; keep only real ones.
	l1, a1 = l1[:min(mid, len(l1))], a1[:min(mid, len(a1))]
	return append(l1, l2...), append(a1, a2...), nil
}

// isNoProfileError reports a shape the engine has no optimisation profile for.
func isNoProfileError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "do not satisfy any optimization profile") ||
		strings.Contains(msg, "does not satisfy an optimization profile")
}

// runChunk executes one forward pass for up to `limit` questions and returns the
// logits and action rows.
func (s *Service) runChunk(ctx context.Context, items []preparedQuestion, padLen int) ([][]float32, [][]float32, error) {
	// Every question in the batch shares one sequence length, so pad to the
	// longest. Padding is safe: the encoder honours attention_mask.
	length := 0
	for _, it := range items {
		if len(it.built.IDs) > length {
			length = len(it.built.IDs)
		}
	}

	// A dynamic engine still has a floor: the profile's min shape. A short input
	// that does not reach it fails with "does not satisfy any optimization
	// profiles", so pad up to the floor rather than sending the natural length.
	if seqMin, seqMax, ok := s.model.InputBounds("input_ids", 1); ok {
		if length > seqMax {
			return nil, nil, fmt.Errorf(
				"inference: 这段输入需要 %d 个 token，但这个 engine 最多接受 %d 个。"+
					"要么缩短输入，要么按更大的序列长度重新编译 engine"+
					"（trtexec --maxShapes=input_ids:1x%d,...）",
				length, seqMax, length)
		}
		if length < seqMin {
			length = seqMin
		}
	}

	// A fixed-shape plan declares one exact length; use it.
	if padLen > 0 {
		if length > padLen {
			return nil, nil, fmt.Errorf(
				"inference: 这段输入需要 %d 个 token，但这个 engine 固定只接受 %d 个。"+
					"engine 是在编译时定死序列长度的，所以要么缩短输入，要么按更长的序列重新编译 engine"+
					"（trtexec --shapes=input_ids:1x%d,...）",
				length, padLen, length)
		}
		length = padLen
	}

	// The head scores one marker per option; the widest question sets K.
	kMax := 0
	for _, it := range items {
		if len(it.built.MarkerPositions) > kMax {
			kMax = len(it.built.MarkerPositions)
		}
	}
	// A fixed-shape plan declares an exact marker count; pad K up to it and let
	// marker_mask mark the real entries.
	if fixed := fixedDim(s.modelFixedShape("marker_pos"), 1); fixed > 0 {
		if kMax > fixed {
			return nil, nil, fmt.Errorf(
				"inference: 问题有 %d 个选项，但这个 engine 一次只能评分 %d 个。"+
					"减少选项数量，或按更大的 marker 维度重新编译 engine"+
					"（trtexec --shapes=marker_pos:1x%d,...）",
				kMax, fixed, kMax)
		}
		kMax = fixed
	}

	batch := len(items)
	// A fixed-shape plan also pins the batch dimension.
	if fixed := fixedDim(s.modelFixedShape("input_ids"), 0); fixed > 0 {
		batch = fixed
	}
	inputIDs := make([]int64, batch*length)
	attMask := make([]int64, batch*length)
	markerPos := make([]int64, batch*kMax)
	markerMask := make([]byte, batch*kMax)
	qtypes := make([]int64, batch)

	for row, it := range items {
		// Write straight into the batch arrays rather than allocating a padded
		// pair per row and copying it in.
		sequence.PadInto(it.built.IDs, inputIDs[row*length:(row+1)*length],
			attMask[row*length:(row+1)*length], s.tok.PadID())
		for i, pos := range it.built.MarkerPositions {
			markerPos[row*kMax+i] = int64(pos)
			markerMask[row*kMax+i] = 1
		}
		qtypes[row] = int64(it.built.QType)
	}

	// One forward pass over the batch. The kernel is named only through the
	// contract, so this code is identical whichever one is resident.
	outputs, err := s.model.Run(ctx, backend.RunInput{
		Inputs: []backend.Tensor{
			{Name: "input_ids", DType: backend.Int64, Data: inputIDs, Shape: []int{batch, length}},
			{Name: "attention_mask", DType: backend.Int64, Data: attMask, Shape: []int{batch, length}},
			{Name: "marker_pos", DType: backend.Int64, Data: markerPos, Shape: []int{batch, kMax}},
			{Name: "marker_mask", DType: backend.Bool, Data: markerMask, Shape: []int{batch, kMax}},
			{Name: "qtype", DType: backend.Int64, Data: qtypes, Shape: []int{batch}},
		},
		Outputs: []string{"logits", "act_logits"},
	})
	if err != nil {
		return nil, nil, err
	}

	var logitsRows, actRows [][]float32
	for i, out := range outputs {
		flat, err := out.Floats()
		if err != nil {
			return nil, nil, err
		}
		rows := splitRows(flat, batch)
		if len(rows) != batch {
			return nil, nil, fmt.Errorf("inference: output %q has %d rows, expected %d",
				out.Name, len(rows), batch)
		}
		// Only the first len(items) rows carry questions; any remaining rows
		// exist because the model pins the batch dimension.
		rows = rows[:len(items)]
		// Dispatch on the name the kernel reports for this index rather than on
		// position: the output order is the model's, not ours.
		switch out.Name {
		case "logits":
			logitsRows = rows
		case "act_logits":
			actRows = rows
		default:
			// An auxiliary output the model does not use.
		}
		if debugInference {
			fmt.Printf("[inference] output[%d] name=%q shape=%v first=%v\n",
				i, out.Name, out.Shape, flat[:min(4, len(flat))])
		}
	}
	return logitsRows, actRows, nil
}

// buildAnswer turns a probability vector into the typed answer for a question.
func buildAnswer(spec QuestionSpec, q sequence.Question, built sequence.Built, probs []float64, actProb float64) Answer {
	ans := Answer{
		Type:       q.Type.String(),
		Confidence: round4(confidenceFromProbs(probs)),
		Action:     Action{ActProbability: round4(actProb)},
	}

	switch q.Type {
	case sequence.Choice:
		labels := q.Labels
		if len(labels) == 0 {
			labels = built.Options
		}
		dist := make(map[string]float64, len(probs))
		best, bestIdx := math.Inf(-1), 0
		for i, p := range probs {
			label := fmt.Sprintf("%d", i)
			if i < len(labels) {
				label = labels[i]
			}
			dist[label] = round4(p)
			if p > best {
				best, bestIdx = p, i
			}
		}
		if bestIdx < len(labels) {
			ans.Choice = labels[bestIdx]
		}
		ans.Probabilities = dist

	case sequence.Score:
		// Expected level on the ordinal rubric.
		expected := 0.0
		dist := make(map[string]float64, len(probs))
		legend := make(map[string]string, len(probs))
		for i, p := range probs {
			expected += float64(i) * p
			dist[fmt.Sprintf("%d", i)] = round4(p)
			if i < len(spec.Levels()) {
				legend[fmt.Sprintf("%d", i)] = spec.Levels()[i]
			}
		}
		ans.Score = round4(expected)
		ans.Probabilities = dist
		ans.Legend = legend

	default: // noul
		pTrue := 0.0
		if len(probs) > 1 {
			pTrue = probs[1]
		}
		ans.NoUL = round4(pTrue)
		ans.NoULSet = true
		ans.Confidence = round4(math.Max(pTrue, 1-pTrue))
	}
	return ans
}

// SerializeState renders a state the way the Python SDK does: a string passes
// through, anything else becomes compact JSON.
func SerializeState(state any) string {
	switch v := state.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(raw)
	}
}

// Levels returns the score question's level descriptions.
func (q QuestionSpec) Levels() []string {
	list, ok := q.Criteria.([]any)
	if !ok {
		if s, ok := q.Criteria.([]string); ok {
			return s
		}
		return nil
	}
	out := make([]string, 0, len(list))
	for _, v := range list {
		out = append(out, renderValue(v))
	}
	return out
}

func renderValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(raw)
}

// BuildQuestion converts an API question into the builder's form, rendering
// option texts exactly as the SDK's render_options does.
func BuildQuestion(spec QuestionSpec) (sequence.Question, error) {
	qt, err := sequence.ParseQType(spec.Type)
	if err != nil {
		return sequence.Question{}, err
	}
	if strings.TrimSpace(spec.Instructions) == "" {
		return sequence.Question{}, fmt.Errorf("instructions are required")
	}
	q := sequence.Question{Type: qt, Instructions: spec.Instructions}

	switch qt {
	case sequence.Choice:
		switch crit := spec.Criteria.(type) {
		case map[string]any:
			// Sort labels for a deterministic option order; the SDK iterates a
			// dict, whose order Go cannot reproduce, and the head is
			// order-sensitive only in that markers must line up with labels.
			labels := make([]string, 0, len(crit))
			for k := range crit {
				labels = append(labels, k)
			}
			sort.Strings(labels)
			for _, k := range labels {
				q.Labels = append(q.Labels, k)
				desc := renderValue(crit[k])
				if desc == "" || desc == `""` {
					q.Criteria = append(q.Criteria, k)
				} else {
					q.Criteria = append(q.Criteria, k+": "+desc)
				}
			}
		case []any:
			for _, c := range crit {
				s := renderValue(c)
				q.Labels = append(q.Labels, s)
				q.Criteria = append(q.Criteria, s)
			}
		default:
			return sequence.Question{}, fmt.Errorf("choice questions need criteria as an object or array")
		}
		if len(q.Criteria) == 0 {
			return sequence.Question{}, fmt.Errorf("choice questions need at least one criterion")
		}

	case sequence.Score:
		levels := spec.Levels()
		if len(levels) == 0 {
			return sequence.Question{}, fmt.Errorf("score questions need criteria as a non-empty array")
		}
		for i, lvl := range levels {
			q.Criteria = append(q.Criteria, fmt.Sprintf("level %d: %s", i, lvl))
		}

	case sequence.NoUL:
		// Criteria is ignored; the builder supplies [false, true].
	}

	return q, nil
}

// actProbability turns the head's action logits into a probability.
//
// The exported model returns raw action logits (observed around ±4700), not a
// softmax, so the raw value is not a probability and exponentiating it naively
// overflows. The Python SDK applies a softmax to the pair; do the same.
//
// A NaN result is a known artefact of fp16 engines: the entropy term inside the
// act head computes p*log(p), and under fp16 a probability that underflows to 0
// yields 0 * -inf = NaN. Only the auxiliary act head is affected; the decision
// logits are finite, which is why this returns 0 rather than failing the call.
func actProbability(logits []float32) float64 {
	if len(logits) == 0 {
		return 0
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0
		}
	}
	if len(logits) == 1 {
		// A single logit is a degenerate distribution; treat it as decided.
		return 1
	}
	maxV := logits[0]
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	sum := 0.0
	pTrue := 0.0
	for i, v := range logits {
		e := math.Exp(float64(v - maxV))
		sum += e
		if i == 0 {
			pTrue = e
		}
	}
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return 0
	}
	p := pTrue / sum
	if math.IsNaN(p) || math.IsInf(p, 0) {
		return 0
	}
	return p
}

// confidenceFromProbs is normalised Shannon entropy: 1 - H(p)/log(k).
func confidenceFromProbs(p []float64) float64 {
	k := len(p)
	if k < 2 {
		return 1
	}
	ent := 0.0
	for _, x := range p {
		if x > 1e-12 {
			ent -= x * math.Log(x)
		}
	}
	c := 1 - ent/math.Log(float64(k))
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// softmaxWithTemperature normalises a logit row after dividing by the fitted
// calibration temperature, matching the SDK's `z = logits / t_scale`.
//
// A scale of 1.0 (or less) is treated as no scaling when it is not finite.
func softmaxWithTemperature(logits []float32, temperature float64) ([]float64, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("inference: empty logits row")
	}
	scale := temperature
	if scale <= 0 || scale != scale {
		scale = 1.0
	}
	maxV := float64(logits[0]) / scale
	for _, v := range logits {
		if z := float64(v) / scale; z > maxV {
			maxV = z
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		e := math.Exp(float64(v)/scale - maxV)
		out[i] = e
		sum += e
	}
	if sum == 0 {
		return nil, fmt.Errorf("inference: logits row is degenerate")
	}
	for i := range out {
		out[i] /= sum
	}
	return out, nil
}

func splitRows(flat []float32, rows int) [][]float32 {
	if rows <= 0 || len(flat) == 0 {
		return nil
	}
	per := len(flat) / rows
	if per == 0 {
		return nil
	}
	out := make([][]float32, 0, rows)
	for i := 0; i < rows; i++ {
		out = append(out, flat[i*per:(i+1)*per])
	}
	return out
}

func round4(v float64) float64 { return math.Round(v*10000) / 10000 }

func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}
