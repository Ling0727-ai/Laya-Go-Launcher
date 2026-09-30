package backend

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DType is a tensor element type, independent of any kernel's own enum.
//
// The numeric values are the ONNX Runtime element-type codes, so the ONNX
// backend maps them straight through and only the TensorRT adapter has to
// translate. They are not part of a public wire format.
type DType int

const (
	Float32  DType = 1
	Uint8    DType = 2
	Int8     DType = 3
	Uint16   DType = 4
	Int16    DType = 5
	Int32    DType = 6
	Int64    DType = 7
	Bool     DType = 9
	Float16  DType = 10
	Float64  DType = 11
	BFloat16 DType = 16
)

// String names the element type as it appears in API payloads.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Float64:
		return "float64"
	case Int64:
		return "int64"
	case Int32:
		return "int32"
	case Int16:
		return "int16"
	case Uint16:
		return "uint16"
	case Int8:
		return "int8"
	case Uint8:
		return "uint8"
	case Bool:
		return "bool"
	default:
		return fmt.Sprintf("dtype(%d)", int(d))
	}
}

// Size is the byte width of one element, or 0 when unknown.
func (d DType) Size() int {
	switch d {
	case Float64, Int64:
		return 8
	case Float32, Int32:
		return 4
	case Float16, BFloat16, Int16, Uint16:
		return 2
	case Int8, Uint8, Bool:
		return 1
	default:
		return 0
	}
}

// TensorInfo describes one model IO tensor.
type TensorInfo struct {
	Name  string
	DType DType
	// Shape is the declared shape. A dimension that is dynamic is reported as
	// -1, which is the convention every kernel here uses.
	Shape   []int
	IsInput bool
}

// NumElements is the element count implied by Shape, or 0 when a dimension is
// dynamic.
func (t TensorInfo) NumElements() int {
	if len(t.Shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range t.Shape {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

// Tensor is a named input with its data.
//
// Data must be the Go slice matching DType: []float32, []int64, []int32,
// []int8, []uint8, []bool or []byte. The kernel copies it during Run, so the
// caller may reuse the slice on return.
type Tensor struct {
	Name  string
	DType DType
	Shape []int
	Data  any
}

// NumElements is the element count implied by Shape.
func (t Tensor) NumElements() int {
	n := 1
	for _, d := range t.Shape {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

// ByteSize is how many bytes Data must hold for Shape and DType.
func (t Tensor) ByteSize() int {
	size := t.DType.Size()
	if size == 0 {
		return 0
	}
	n := t.NumElements()
	if n == 0 {
		return 0
	}
	return n * size
}

// Validate checks that Data matches Shape and DType, so a mismatch is reported
// before it reaches a native call that would read out of bounds.
func (t Tensor) Validate() error {
	if t.Name == "" {
		return fmt.Errorf("backend: tensor name is required")
	}
	if t.DType.Size() == 0 {
		return fmt.Errorf("backend: tensor %q has unsupported dtype %s", t.Name, t.DType)
	}
	if len(t.Shape) == 0 {
		return fmt.Errorf("backend: tensor %q needs at least one dimension", t.Name)
	}
	want := t.NumElements()
	if want == 0 {
		return fmt.Errorf("backend: tensor %q has a non-positive dimension in %v", t.Name, t.Shape)
	}
	got, err := elementCount(t.Data, t.DType)
	if err != nil {
		return fmt.Errorf("backend: tensor %q: %w", t.Name, err)
	}
	if got < want {
		return fmt.Errorf("backend: tensor %q needs %d elements for %v, got %d",
			t.Name, want, t.Shape, got)
	}
	return nil
}

// elementCount returns the element count of a typed slice and checks that its
// element type matches the declared dtype.
//
// []byte is accepted for Bool because Go's bool slice is one byte per element
// and the tokenizer's mask pipeline produces []byte; both are 1-byte elements.
func elementCount(data any, dtype DType) (int, error) {
	switch v := data.(type) {
	case []float32:
		if dtype != Float32 {
			return 0, fmt.Errorf("dtype %s does not match []float32", dtype)
		}
		return len(v), nil
	case []float64:
		if dtype != Float64 {
			return 0, fmt.Errorf("dtype %s does not match []float64", dtype)
		}
		return len(v), nil
	case []int64:
		if dtype != Int64 {
			return 0, fmt.Errorf("dtype %s does not match []int64", dtype)
		}
		return len(v), nil
	case []int32:
		if dtype != Int32 {
			return 0, fmt.Errorf("dtype %s does not match []int32", dtype)
		}
		return len(v), nil
	case []int16:
		if dtype != Int16 {
			return 0, fmt.Errorf("dtype %s does not match []int16", dtype)
		}
		return len(v), nil
	case []uint16:
		if dtype != Uint16 {
			return 0, fmt.Errorf("dtype %s does not match []uint16", dtype)
		}
		return len(v), nil
	case []int8:
		if dtype != Int8 {
			return 0, fmt.Errorf("dtype %s does not match []int8", dtype)
		}
		return len(v), nil
	case []uint8:
		if dtype != Uint8 && dtype != Bool {
			return 0, fmt.Errorf("dtype %s does not match []uint8", dtype)
		}
		return len(v), nil
	case []bool:
		if dtype != Bool {
			return 0, fmt.Errorf("dtype %s does not match []bool", dtype)
		}
		return len(v), nil
	default:
		return 0, fmt.Errorf("unsupported slice type %T", data)
	}
}

// Output is one tensor read back from a run.
//
// Data is raw little-endian bytes so any element type is expressible; use
// Floats for the decision head's float32 outputs.
type Output struct {
	Name  string
	DType DType
	Shape []int
	Data  []byte
}

// Floats interprets Data as float32, which is what the laya decision head and
// act head both emit.
func (o Output) Floats() ([]float32, error) {
	if o.DType != Float32 {
		return nil, fmt.Errorf("backend: output %q is %s, not float32", o.Name, o.DType)
	}
	if len(o.Data)%4 != 0 {
		return nil, fmt.Errorf("backend: output %q is not a whole number of float32", o.Name)
	}
	out := make([]float32, len(o.Data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(o.Data[i*4:]))
	}
	return out, nil
}

// Float32Bytes renders a float32 slice as little-endian bytes, for tests and
// for backends that carry outputs as raw memory.
func Float32Bytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}
