package backend

import "testing"

// TestParseKind pins the accepted kernel names and the aliases, because this is
// what a config file, an environment variable and a CLI flag all funnel through.
func TestParseKind(t *testing.T) {
	cases := []struct {
		in      string
		want    Kind
		wantErr bool
	}{
		{"", KindAuto, false},
		{"auto", KindAuto, false},
		{"AUTO", KindAuto, false},
		{"  auto  ", KindAuto, false},
		{"tensorrt", KindTensorRT, false},
		{"TensorRT", KindTensorRT, false},
		{"trt", KindTensorRT, false},
		{"plan", KindTensorRT, false},
		{"engine", KindTensorRT, false},
		{"onnx", KindONNX, false},
		{"ONNX", KindONNX, false},
		{"onnxruntime", KindONNX, false},
		{"ort", KindONNX, false},
		{"cuda", "", true},
		{"directml", "", true},
		{"onnx-cpu", "", true},
		{"nonsense", "", true},
	}
	for _, c := range cases {
		got, err := ParseKind(c.in)
		if c.wantErr {
			if err == nil {
				t.Errorf("ParseKind(%q) returned %q, want an error", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseKind(%q): unexpected error %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("ParseKind(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestEmptyKindIsAuto guards the compatibility property that matters most: a
// config file written before the backend option existed has no "backend" key,
// and must keep meaning what it meant — use the TensorRT path.
func TestEmptyKindIsAuto(t *testing.T) {
	got, err := ParseKind("")
	if err != nil {
		t.Fatalf("ParseKind(\"\") errored: %v", err)
	}
	if got != KindAuto {
		t.Fatalf("ParseKind(\"\") = %q, want auto", got)
	}
}

// TestFindAndInputNames covers the small helpers the factory and the use case
// rely on for tensor lookup.
func TestFindAndInputNames(t *testing.T) {
	info := Info{
		Inputs: []TensorInfo{
			{Name: "input_ids", DType: Int64, Shape: []int{1, -1}, IsInput: true},
			{Name: "qtype", DType: Int64, Shape: []int{1}, IsInput: true},
		},
		Outputs: []TensorInfo{
			{Name: "logits", DType: Float32, Shape: []int{-1, -1}},
		},
	}

	if names := info.InputNames(); len(names) != 2 || names[0] != "input_ids" || names[1] != "qtype" {
		t.Errorf("InputNames() = %v", names)
	}
	if names := info.OutputNames(); len(names) != 1 || names[0] != "logits" {
		t.Errorf("OutputNames() = %v", names)
	}

	if got, ok := Find(info.Inputs, "qtype"); !ok || got.DType != Int64 {
		t.Errorf("Find(qtype) = %+v, %v", got, ok)
	}
	if _, ok := Find(info.Inputs, "nope"); ok {
		t.Error("Find(nope) reported a match")
	}
}

// TestRunInputLookup covers the accessor the ONNX path uses to reach a tensor.
func TestRunInputLookup(t *testing.T) {
	in := RunInput{Inputs: []Tensor{
		{Name: "input_ids", DType: Int64, Shape: []int{1, 4}, Data: []int64{1, 2, 3, 4}},
	}}
	got, ok := in.Input("input_ids")
	if !ok {
		t.Fatal("Input(input_ids) not found")
	}
	if got.DType != Int64 {
		t.Errorf("dtype = %s, want int64", got.DType)
	}
	if _, ok := in.Input("missing"); ok {
		t.Error("Input(missing) reported a match")
	}
}
