package inference

import "testing"

// TestChooseMaxBatch pins the batching decision.
//
// The bug this guards against is subtle and was live: a dynamic batch dimension
// is reported as -1 in the engine's shape, so a check of the form "shape[0] > 1"
// answered 1 for every dynamic-batch plan and disabled batching entirely. The
// caller then ran one forward pass per question and paid the per-run fixed cost
// each time, which is the dominant cost for this model.
func TestChooseMaxBatch(t *testing.T) {
	cases := []struct {
		name    string
		shape   []int
		profMax int
		profOK  bool
		want    int
	}{
		{
			name:  "dynamic batch with a profile range",
			shape: []int{-1, -1}, profMax: 8, profOK: true,
			want: 8,
		},
		{
			name:  "dynamic batch capped by maxBatchCap",
			shape: []int{-1, -1}, profMax: 256, profOK: true,
			want: maxBatchCap,
		},
		{
			// An ONNX graph: the batch dimension is symbolic and has no
			// profile, so it accepts any batch up to the cap.
			name:  "symbolic batch with no profile (onnx)",
			shape: []int{-1, -1}, profMax: 0, profOK: false,
			want: maxBatchCap,
		},
		{
			name:  "dynamic batch whose profile max is 1",
			shape: []int{-1, -1}, profMax: 1, profOK: true,
			want: 1,
		},
		{
			name:  "fixed batch of 1 has no usable range",
			shape: []int{1, -1}, profMax: 0, profOK: false,
			want: 1,
		},
		{
			// A plan fixed at batch 1 reports min == max == 1 through
			// InputBounds, so profMax is 1 and profOK is true.
			name:  "fixed batch of 1 reported through the profile",
			shape: []int{1, -1}, profMax: 1, profOK: true,
			want: 1,
		},
		{
			name:  "fixed batch above 1 is the exact size",
			shape: []int{4, -1}, profMax: 4, profOK: true,
			want: 4,
		},
		{
			name:  "fixed batch above 1 without profile info",
			shape: []int{3, 512}, profMax: 0, profOK: false,
			want: 3,
		},
		{
			name:  "unknown shape falls back to the profile",
			shape: nil, profMax: 6, profOK: true,
			want: 6,
		},
		{
			name:  "unknown shape and no profile",
			shape: nil, profMax: 0, profOK: false,
			want: 1,
		},
		{
			name:  "batch dimension 2",
			shape: []int{-1, -1}, profMax: 2, profOK: true,
			want: 2,
		},
		{
			name:  "empty shape slice",
			shape: []int{}, profMax: 4, profOK: true,
			want: 4,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := chooseMaxBatch(c.shape, c.profMax, c.profOK)
			if got != c.want {
				t.Errorf("chooseMaxBatch(%v, profMax=%d, profOK=%v) = %d, want %d",
					c.shape, c.profMax, c.profOK, got, c.want)
			}
			if got < 1 {
				t.Errorf("batch size must be at least 1, got %d", got)
			}
		})
	}
}

// TestMaxBatchCapIsSane keeps the cap from being raised without thought: the
// cap exists so a request with a couple of questions is not padded into a huge
// batch whose padding rows cost more than the saved forward passes.
func TestMaxBatchCapIsSane(t *testing.T) {
	if maxBatchCap < 2 {
		t.Fatalf("maxBatchCap = %d: batching would never engage", maxBatchCap)
	}
	if maxBatchCap > 64 {
		t.Errorf("maxBatchCap = %d: large enough that padding could dominate", maxBatchCap)
	}
}
