package backend

import (
	"math"
	"testing"
)

// TestDTypeSize pins the byte widths, because a wrong width silently truncates
// a tensor or reads past its end.
func TestDTypeSize(t *testing.T) {
	cases := []struct {
		dtype DType
		want  int
	}{
		{Float32, 4},
		{Float64, 8},
		{Float16, 2},
		{BFloat16, 2},
		{Int64, 8},
		{Int32, 4},
		{Int16, 2},
		{Uint16, 2},
		{Int8, 1},
		{Uint8, 1},
		{Bool, 1},
		{DType(999), 0},
	}
	for _, c := range cases {
		if got := c.dtype.Size(); got != c.want {
			t.Errorf("%s.Size() = %d, want %d", c.dtype, got, c.want)
		}
	}
}

// TestDTypeString checks the names that appear in API payloads.
func TestDTypeString(t *testing.T) {
	cases := map[DType]string{
		Float32: "float32",
		Float16: "float16",
		Int64:   "int64",
		Bool:    "bool",
	}
	for dtype, want := range cases {
		if got := dtype.String(); got != want {
			t.Errorf("DType(%d).String() = %q, want %q", dtype, got, want)
		}
	}
}

// TestTensorValidateAcceptsTheRealContract covers exactly the five tensors the
// predict use case builds, including the []byte carrier for a bool mask.
func TestTensorValidateAcceptsTheRealContract(t *testing.T) {
	cases := []struct {
		name   string
		tensor Tensor
	}{
		{"input_ids", Tensor{
			Name: "input_ids", DType: Int64,
			Data: []int64{1, 2, 3, 4}, Shape: []int{1, 4},
		}},
		{"attention_mask", Tensor{
			Name: "attention_mask", DType: Int64,
			Data: []int64{1, 1, 1, 1}, Shape: []int{1, 4},
		}},
		{"marker_pos", Tensor{
			Name: "marker_pos", DType: Int64,
			Data: []int64{1, 2}, Shape: []int{1, 2},
		}},
		{"marker_mask as bytes", Tensor{
			Name: "marker_mask", DType: Bool,
			Data: []byte{1, 1}, Shape: []int{1, 2},
		}},
		{"marker_mask as bools", Tensor{
			Name: "marker_mask", DType: Bool,
			Data: []bool{true, true}, Shape: []int{1, 2},
		}},
		{"qtype", Tensor{
			Name: "qtype", DType: Int64,
			Data: []int64{0}, Shape: []int{1},
		}},
	}
	for _, c := range cases {
		if err := c.tensor.Validate(); err != nil {
			t.Errorf("%s: Validate() = %v, want nil", c.name, err)
		}
	}
}

// TestTensorValidateRejectsMismatches is the guard that keeps a bad shape or
// dtype from reaching a native call that would read out of bounds.
func TestTensorValidateRejectsMismatches(t *testing.T) {
	cases := []struct {
		name   string
		tensor Tensor
	}{
		{"no name", Tensor{Name: "", DType: Int64, Data: []int64{1}, Shape: []int{1}}},
		{"no shape", Tensor{Name: "x", DType: Int64, Data: []int64{1}, Shape: nil}},
		{"zero dim", Tensor{Name: "x", DType: Int64, Data: []int64{1}, Shape: []int{0}}},
		{"negative dim", Tensor{Name: "x", DType: Int64, Data: []int64{1}, Shape: []int{-1}}},
		{"unsupported dtype", Tensor{Name: "x", DType: DType(999), Data: []int64{1}, Shape: []int{1}}},
		{"wrong slice type", Tensor{Name: "x", DType: Int64, Data: []float32{1}, Shape: []int{1}}},
		{"too few elements", Tensor{Name: "x", DType: Int64, Data: []int64{1}, Shape: []int{1, 4}}},
		{"unsupported data", Tensor{Name: "x", DType: Int64, Data: "nope", Shape: []int{1}}},
		{"float32 as int64", Tensor{Name: "x", DType: Int64, Data: []float32{1}, Shape: []int{1}}},
		{"byte as float32", Tensor{Name: "x", DType: Float32, Data: []byte{1}, Shape: []int{1}}},
	}
	for _, c := range cases {
		if err := c.tensor.Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want an error", c.name)
		}
	}
}

// TestTensorByteSize checks the size the ONNX bridge uses to describe a buffer.
func TestTensorByteSize(t *testing.T) {
	cases := []struct {
		tensor Tensor
		want   int
	}{
		{Tensor{DType: Int64, Shape: []int{2, 3}}, 48},
		{Tensor{DType: Float32, Shape: []int{1, 4}}, 16},
		{Tensor{DType: Bool, Shape: []int{5}}, 5},
		{Tensor{DType: Int64, Shape: []int{-1, 3}}, 0},
		{Tensor{DType: DType(999), Shape: []int{3}}, 0},
	}
	for _, c := range cases {
		if got := c.tensor.ByteSize(); got != c.want {
			t.Errorf("%+v.ByteSize() = %d, want %d", c.tensor, got, c.want)
		}
	}
}

// TestTensorInfoNumElements checks that a dynamic dimension reports 0 elements
// rather than a wrong count.
func TestTensorInfoNumElements(t *testing.T) {
	cases := []struct {
		info TensorInfo
		want int
	}{
		{TensorInfo{Shape: []int{1, 4}}, 4},
		{TensorInfo{Shape: []int{2, 3, 4}}, 24},
		{TensorInfo{Shape: []int{1, -1}}, 0},
		{TensorInfo{Shape: []int{-1, -1}}, 0},
		{TensorInfo{Shape: nil}, 0},
	}
	for _, c := range cases {
		if got := c.info.NumElements(); got != c.want {
			t.Errorf("%v.NumElements() = %d, want %d", c.info.Shape, got, c.want)
		}
	}
}

// TestOutputFloats checks the float32 reader the predict use case depends on.
func TestOutputFloats(t *testing.T) {
	want := []float32{0.25, -1.5, 3.75}
	out := Output{Name: "logits", DType: Float32, Data: Float32Bytes(want)}
	got, err := out.Floats()
	if err != nil {
		t.Fatalf("Floats(): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("value %d = %v, want %v", i, got[i], want[i])
		}
	}

	// NaN and ±Inf must survive the round trip: the act head is known to emit
	// NaN under fp16, and actProbability has to see it to handle it.
	special := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	gotSpecial, err := Output{DType: Float32, Data: Float32Bytes(special)}.Floats()
	if err != nil {
		t.Fatalf("Floats() on special values: %v", err)
	}
	if !math.IsNaN(float64(gotSpecial[0])) {
		t.Error("NaN did not survive the round trip")
	}
	if !math.IsInf(float64(gotSpecial[1]), 1) || !math.IsInf(float64(gotSpecial[2]), -1) {
		t.Error("±Inf did not survive the round trip")
	}
}

// TestOutputFloatsRejectsNonFloat32 checks the reader refuses a tensor it cannot
// interpret, instead of reinterpreting bytes as floats.
func TestOutputFloatsRejectsNonFloat32(t *testing.T) {
	int64Out := Output{Name: "x", DType: Int64, Data: make([]byte, 8)}
	if _, err := int64Out.Floats(); err == nil {
		t.Error("Floats() on an int64 output returned no error")
	}

	partial := Output{Name: "x", DType: Float32, Data: make([]byte, 3)}
	if _, err := partial.Floats(); err == nil {
		t.Error("Floats() on a partial float32 returned no error")
	}
}
