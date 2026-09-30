// Package metrics records request counters and a bounded latency ring.
//
// The ring is deliberately fixed-size: an uptime of weeks must not grow the
// process's memory, and percentiles over the last few hundred requests are what
// an operator actually looks at.
package metrics

import (
	"sort"
	"sync"
	"time"
)

// ringSize bounds the latency sample ring.
const ringSize = 512

// Recorder is safe for concurrent use.
type Recorder struct {
	mu sync.Mutex

	requestsTotal  int64
	requestsFailed int64
	predictTotal   int64
	engineLoads    int64

	ring   [ringSize]float64
	filled int
	next   int

	lastRun    time.Time
	lastTiming struct {
		TotalMS     float64
		TokenizeMS  float64
		InferenceMS float64
	}
}

// New returns an empty recorder.
func New() *Recorder { return &Recorder{} }

// RecordEngineLoad notes a successful load.
func (r *Recorder) RecordEngineLoad() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.engineLoads++
	r.requestsTotal++
}

// RecordFailure notes a failed request.
func (r *Recorder) RecordFailure() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requestsFailed++
	r.requestsTotal++
}

// RecordPredict notes a completed prediction. wall is the observed round trip;
// total/tokenize/inference come from the use case's own breakdown.
func (r *Recorder) RecordPredict(wall time.Duration, totalMS, tokenizeMS, inferenceMS float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.predictTotal++
	r.requestsTotal++

	r.ring[r.next] = float64(wall.Microseconds()) / 1000.0
	r.next = (r.next + 1) % ringSize
	if r.filled < ringSize {
		r.filled++
	}

	r.lastRun = time.Now().UTC()
	r.lastTiming.TotalMS = totalMS
	r.lastTiming.TokenizeMS = tokenizeMS
	r.lastTiming.InferenceMS = inferenceMS
}

// Latency is the percentile block of a snapshot.
type Latency struct {
	P50 float64 `json:"p50"`
	P90 float64 `json:"p90"`
	P99 float64 `json:"p99"`
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// LastRun describes the most recent prediction.
type LastRun struct {
	At          string  `json:"at"`
	TotalMS     float64 `json:"total_ms"`
	TokenizeMS  float64 `json:"tokenize_ms"`
	InferenceMS float64 `json:"inference_ms"`
}

// Snapshot is the serialisable form.
type Snapshot struct {
	RequestsTotal  int64    `json:"requests_total"`
	RequestsFailed int64    `json:"requests_failed"`
	PredictTotal   int64    `json:"predict_total"`
	EngineLoads    int64    `json:"engine_loads"`
	Samples        int      `json:"latency_samples"`
	Latency        Latency  `json:"latency_ms"`
	LastRun        *LastRun `json:"last_run,omitempty"`
}

// Snapshot reads the current counters.
func (r *Recorder) Snapshot() Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	s := Snapshot{
		RequestsTotal:  r.requestsTotal,
		RequestsFailed: r.requestsFailed,
		PredictTotal:   r.predictTotal,
		EngineLoads:    r.engineLoads,
		Samples:        r.filled,
	}
	if r.filled > 0 {
		samples := make([]float64, r.filled)
		copy(samples, r.ring[:r.filled])
		sort.Float64s(samples)
		s.Latency = Latency{
			P50: percentile(samples, 0.50),
			P90: percentile(samples, 0.90),
			P99: percentile(samples, 0.99),
			Min: samples[0],
			Max: samples[len(samples)-1],
		}
	}
	if !r.lastRun.IsZero() {
		s.LastRun = &LastRun{
			At:          r.lastRun.Format(time.RFC3339),
			TotalMS:     r.lastTiming.TotalMS,
			TokenizeMS:  r.lastTiming.TokenizeMS,
			InferenceMS: r.lastTiming.InferenceMS,
		}
	}
	return s
}

// percentile picks the nearest-rank value from a sorted slice.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
