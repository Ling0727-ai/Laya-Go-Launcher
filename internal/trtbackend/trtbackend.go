// Package trtbackend implements backend.Backend over the TensorRT plan manager.
//
// This is the historical path, unchanged in behaviour: a plan is loaded through
// internal/kernel, a pool of execution contexts is pre-allocated, and a run
// borrows one. Everything TensorRT-specific — the context pool, the VRAM-derived
// context count, the activation-memory accounting — stays inside this package,
// so the use case above it never sees a plan, a context or a CUDA stream.
package trtbackend

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/engine"
	"github.com/local/laya-go-launcher/internal/kernel"
)

// Backend serves models through TensorRT.
type Backend struct {
	mgr *engine.Manager
}

// New builds a backend over a fresh manager.
func New() *Backend { return &Backend{mgr: engine.NewManager()} }

// NewWithManager builds a backend over an existing manager.
//
// The manager is exposed through Manager so the bench and probe commands can
// keep using the TensorRT-specific surface directly.
func NewWithManager(mgr *engine.Manager) *Backend { return &Backend{mgr: mgr} }

// Kind names the kernel.
func (b *Backend) Kind() backend.Kind { return backend.KindTensorRT }

// Initialize brings up the TensorRT and CUDA runtime.
//
// Exposed so the factory can probe availability without loading a plan.
func Initialize() error { return kernel.Initialize() }

// RuntimeVersion is the TensorRT version the process is linked against.
func RuntimeVersion() string { return kernel.Version() }

// DeviceInfo reports the active CUDA device.
func DeviceInfo() kernel.DeviceInfo { return kernel.QueryDeviceInfo() }

// VRAMInfo reports device memory in MiB.
func VRAMInfo() kernel.VRAMInfo { return kernel.QueryVRAMInfo() }

// SetDiagnostics turns the native TensorRT diagnostic log on or off.
func SetDiagnostics(enabled bool) { kernel.SetDiagnostics(enabled) }

// Manager exposes the underlying plan manager.
func (b *Backend) Manager() *engine.Manager { return b.mgr }

// Load deserialises a plan and pre-allocates its contexts.
//
// A .onnx path is refused before the native call, for the same reason the ONNX
// kernel refuses a .engine: the two formats are not interchangeable, so the
// useful answer is "wrong kernel", not "failed to deserialize" after reading a
// multi-gigabyte graph into memory to find out.
func (b *Backend) Load(ctx context.Context, opts backend.Options) (backend.Info, error) {
	if strings.EqualFold(filepath.Ext(opts.Path), ".onnx") {
		return backend.Info{}, fmt.Errorf(
			"%w: %s is an ONNX graph; select the onnx kernel for it",
			backend.ErrWrongFormat, opts.Path)
	}
	info, err := b.mgr.Load(engine.Options{Path: opts.Path, Contexts: opts.Contexts})
	if err != nil {
		return backend.Info{}, backend.WrapAllocation(translate(err))
	}
	return describe(info), nil
}

// Unload releases the plan.
func (b *Backend) Unload() { b.mgr.Unload() }

// Close releases everything.
func (b *Backend) Close() { b.mgr.Close() }

// Loaded reports whether a plan is resident.
func (b *Backend) Loaded() bool { return b.mgr.Loaded() }

// Info describes the resident plan.
func (b *Backend) Info() (backend.Info, bool) {
	info, ok := b.mgr.Info()
	if !ok {
		return backend.Info{}, false
	}
	return describe(info), true
}

// InputBounds reports the range one input dimension accepts.
func (b *Backend) InputBounds(name string, dim int) (int, int, bool) {
	return b.mgr.InputBounds(name, dim)
}

// Dir is the directory the plan came from.
func (b *Backend) Dir() string { return b.mgr.Dir() }

// Run executes one forward pass through a pooled context.
//
// The TensorRT run builder owns its descriptors and staging, so the Go side only
// passes names, dtypes and shapes; the caller's slices are copied during
// AddInput and may be reused on return.
func (b *Backend) Run(ctx context.Context, in backend.RunInput) ([]backend.Output, error) {
	var out []backend.Output
	err := b.mgr.WithContext(ctx, func(c *kernel.Context) error {
		run, err := c.NewRun()
		if err != nil {
			return err
		}
		defer run.Close()

		for _, t := range in.Inputs {
			if err := run.AddInput(t.Name, toKernelDType(t.DType), t.Data, t.Shape); err != nil {
				return err
			}
		}

		// An empty list means every output the model declares.
		want := in.Outputs
		if len(want) == 0 {
			info, ok := b.Info()
			if !ok {
				return backend.ErrNotLoaded
			}
			want = info.OutputNames()
		}
		for _, name := range want {
			if err := run.AddOutput(name); err != nil {
				return err
			}
		}

		infos, err := run.Resolve()
		if err != nil {
			return err
		}
		if err := run.Execute(); err != nil {
			return err
		}

		out = make([]backend.Output, 0, len(infos))
		for i, info := range infos {
			raw, err := run.ReadOutputBytes(i)
			if err != nil {
				return err
			}
			out = append(out, backend.Output{
				Name:  info.Name,
				DType: fromKernelDType(info.DType),
				Shape: info.Shape,
				Data:  raw,
			})
		}
		return nil
	})
	if err != nil {
		return nil, translate(err)
	}
	return out, nil
}

// describe converts a manager description into the neutral one.
func describe(info engine.Info) backend.Info {
	out := backend.Info{
		Path:               info.Path,
		Backend:            "TensorRT",
		Contexts:           info.Contexts,
		ActivationMemoryMB: info.ActivationMemoryMB,
		Runtime:            kernel.Version(),
	}
	for _, t := range info.Inputs {
		out.Inputs = append(out.Inputs, convert(t))
	}
	for _, t := range info.Outputs {
		out.Outputs = append(out.Outputs, convert(t))
	}
	if dev := kernel.QueryDeviceInfo(); dev.Name != "" {
		out.Device = dev.Name
	}
	return out
}

func convert(t kernel.TensorInfo) backend.TensorInfo {
	return backend.TensorInfo{
		Name:    t.Name,
		DType:   fromKernelDType(t.DType),
		Shape:   t.Shape,
		IsInput: t.IsInput,
	}
}

// fromKernelDType maps the kernel's enum onto the neutral one.
//
// The two disagree on ordering (the kernel groups float16/bfloat16 after
// float32; the neutral type uses the ONNX codes), so this is a table rather
// than a cast.
func fromKernelDType(d kernel.DType) backend.DType {
	switch d {
	case kernel.Float32:
		return backend.Float32
	case kernel.Float16:
		return backend.Float16
	case kernel.BFloat16:
		return backend.BFloat16
	case kernel.Int32:
		return backend.Int32
	case kernel.Int64:
		return backend.Int64
	case kernel.Bool:
		return backend.Bool
	case kernel.Int8:
		return backend.Int8
	case kernel.Uint8:
		return backend.Uint8
	default:
		return backend.DType(d)
	}
}

// toKernelDType is the inverse of fromKernelDType.
func toKernelDType(d backend.DType) kernel.DType {
	switch d {
	case backend.Float32:
		return kernel.Float32
	case backend.Float16:
		return kernel.Float16
	case backend.BFloat16:
		return kernel.BFloat16
	case backend.Int32:
		return kernel.Int32
	case backend.Int64:
		return kernel.Int64
	case backend.Bool:
		return kernel.Bool
	case backend.Int8:
		return kernel.Int8
	case backend.Uint8:
		return kernel.Uint8
	default:
		return kernel.DType(d)
	}
}

// compile-time check that the TensorRT path satisfies the contract.
var _ backend.Backend = (*Backend)(nil)

// translate maps the manager's own sentinel errors onto the neutral ones, so a
// caller classifies a failure without knowing which kernel produced it.
func translate(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, engine.ErrNotLoaded) {
		return fmt.Errorf("%w: %v", backend.ErrNotLoaded, err)
	}
	if errors.Is(err, engine.ErrClosed) {
		return fmt.Errorf("%w: %v", backend.ErrClosed, err)
	}
	return err
}
