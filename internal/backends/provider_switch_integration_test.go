package backends

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// TestManualProviderSwitch exercises both ORT distributions in one process.
// It is opt-in because CI may not provide either GPU runtime or the large graph.
func TestManualProviderSwitch(t *testing.T) {
	root, model := os.Getenv("LAYA_TRT_TEST_ONNX_ROOT"), os.Getenv("LAYA_TRT_TEST_ONNX_MODEL")
	if root == "" || model == "" {
		t.Skip("set LAYA_TRT_TEST_ONNX_ROOT and LAYA_TRT_TEST_ONNX_MODEL")
	}
	t.Setenv("LAYA_TRT_ONNX_ROOT", root)
	bridge, err := filepath.Abs(filepath.Join("..", "..", "build", "bin", "Release", "layatrt_onnx.dll"))
	if err != nil {
		t.Fatal(err)
	}
	s := NewSwitcher(Config{Kind: backend.KindAuto, DLLPath: bridge})
	defer s.Close()
	for _, provider := range []string{"cuda", "directml", "cuda"} {
		info, _, err := s.LoadWith(context.Background(), s.Config(), backend.Options{
			Path: model, Kind: backend.KindONNX, Provider: provider,
		})
		if err != nil {
			t.Fatalf("switch to %s: %v", provider, err)
		}
		if !strings.EqualFold(info.Device, provider) {
			t.Fatalf("switch to %s loaded device %q", provider, info.Device)
		}
		t.Logf("provider=%s runtime=%s", info.Device, info.Runtime)
	}
	if plan := os.Getenv("LAYA_TRT_TEST_ENGINE"); plan != "" {
		info, _, err := s.LoadWith(context.Background(), s.Config(), backend.Options{
			Path: plan, Kind: backend.KindTensorRT,
		})
		if err != nil {
			t.Fatalf("switch to TensorRT: %v", err)
		}
		if info.Backend != "TensorRT" {
			t.Fatalf("TensorRT selection loaded %q", info.Backend)
		}
	}
}
