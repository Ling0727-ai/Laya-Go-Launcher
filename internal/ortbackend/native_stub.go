//go:build !windows

package ortbackend

import (
	"context"
	"fmt"

	"github.com/local/laya-go-launcher/internal/backend"
)

// On platforms other than Windows the ONNX bridge is not built, so this stub
// keeps the package compiling and reports a clear reason instead of failing to
// link. It exists so `go build ./...` and `go vet ./...` work anywhere.

type nativeSession struct{}

type nativeConfig struct {
	dllPath         string
	RuntimePath     string
	ModelPath       string
	Provider        string
	ProviderOptions map[string]string
}

func newNativeSession(nativeConfig) (*nativeSession, error) { return nil, errUnavailable() }

func destroyNativeSession(*nativeSession) {}

func (s *nativeSession) inputs() ([]backend.TensorInfo, error)  { return nil, errUnavailable() }
func (s *nativeSession) outputs() ([]backend.TensorInfo, error) { return nil, errUnavailable() }

func (s *nativeSession) run(context.Context, []backend.Tensor, []string) error {
	return errUnavailable()
}

func (s *nativeSession) results() ([]resultInfo, error) { return nil, errUnavailable() }
func (s *nativeSession) copyResult(int, []byte) error   { return errUnavailable() }
func (s *nativeSession) runtimeVersion() string         { return "" }
func (s *nativeSession) providerName() string           { return "" }
func (s *nativeSession) providerNote() string           { return "" }

// RuntimeKind names one distribution of ONNX Runtime. The Windows build resolves
// between several; elsewhere there is nothing to choose between, so the type
// still exists to keep the package's API uniform.
type RuntimeKind string

const (
	RuntimeCUDA13   RuntimeKind = "cuda13"
	RuntimeCUDA12   RuntimeKind = "cuda12"
	RuntimeDirectML RuntimeKind = "directml"
	RuntimeCPU      RuntimeKind = "cpu"
	RuntimeAny      RuntimeKind = "any"
)

// resolveRuntime reports that no runtime can be resolved off Windows, where the
// bridge is not built at all.
func resolveRuntime(explicit, provider string) (string, RuntimeKind, error) {
	return "", RuntimeAny, errUnavailable()
}

// Probe reports that the bridge is unavailable on this platform.
func Probe(string, string) (string, error) { return "", errUnavailable() }

// ProbeFor is Probe; there is no provider to take into account off Windows.
func ProbeFor(string, string, string) (string, error) { return "", errUnavailable() }

func errUnavailable() error {
	return fmt.Errorf("%w: the ONNX Runtime bridge is built on Windows only",
		backend.ErrRuntimeMissing)
}
