package ortbackend

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

func TestCUDASession(t *testing.T) {
	runtimePath := os.Getenv("LAYA_ORT_CUDA_RUNTIME")
	modelPath := os.Getenv("LAYA_ORT_CUDA_MODEL")
	if runtimePath == "" || modelPath == "" {
		t.Skip("set LAYA_ORT_CUDA_RUNTIME and LAYA_ORT_CUDA_MODEL")
	}
	be := New(findBridge(t))
	defer be.Close()
	info, err := be.Load(context.Background(), backend.Options{
		Path: modelPath, RuntimePath: runtimePath, Provider: "cuda",
	})
	if err != nil {
		t.Fatalf("CUDA session: %v", err)
	}
	if !strings.EqualFold(info.Device, "cuda") {
		t.Fatalf("CUDA selected but device = %q", info.Device)
	}
	t.Logf("runtime=%s device=%s", info.Runtime, info.Device)
}

// TestDirectMLSession verifies the installed DirectML distribution through the
// real bridge when test fixtures are supplied. This is opt-in because CI does
// not necessarily have a DirectML runtime or a compatible display adapter.
func TestDirectMLSession(t *testing.T) {
	runtimePath := os.Getenv("LAYA_ORT_DML_RUNTIME")
	modelPath := os.Getenv("LAYA_ORT_DML_MODEL")
	if runtimePath == "" || modelPath == "" {
		t.Skip("set LAYA_ORT_DML_RUNTIME and LAYA_ORT_DML_MODEL")
	}
	be := New(findBridge(t))
	defer be.Close()
	info, err := be.Load(context.Background(), backend.Options{
		Path: modelPath, RuntimePath: runtimePath, Provider: "directml",
	})
	if err != nil {
		t.Fatalf("DirectML session: %v", err)
	}
	if !strings.EqualFold(info.Device, "directml") {
		t.Fatalf("DirectML selected but device = %q", info.Device)
	}
	t.Logf("runtime=%s device=%s", info.Runtime, info.Device)
}
