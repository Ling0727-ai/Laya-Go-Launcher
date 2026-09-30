//go:build windows

package ortbackend

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRuntimeCandidatesPreferProviderPackage(t *testing.T) {
	for _, kind := range []RuntimeKind{RuntimeCUDA13, RuntimeCUDA12, RuntimeDirectML} {
		paths := runtimeCandidates(`C:\runtime-root`, runtimeSpec{kind: kind, main: "onnxruntime.dll"})
		if len(paths) == 0 {
			t.Fatalf("%s: no runtime candidates", kind)
		}
		if !strings.Contains(paths[0], filepath.Join("Assets", "onnx")) {
			t.Errorf("%s: first runtime candidate %q is not provider-specific", kind, paths[0])
		}
	}
}

// TestResolveRuntimeGPUProviderIsStrict checks that an explicit GPU provider
// never resolves a runtime without its provider library (e.g. a CPU build).
func TestResolveRuntimeGPUProviderIsStrict(t *testing.T) {
	root := t.TempDir()
	cpuDir := filepath.Join(root, "Assets", "onnx", "cuda12", "pkg", "lib")
	if err := os.MkdirAll(cpuDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cpuDir, "onnxruntime.dll"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LAYA_TRT_ONNX_ROOT", root)
	t.Setenv("PATH", root)
	path, _, err := resolveRuntime("", "cuda")
	if err == nil && strings.EqualFold(path, filepath.Join(cpuDir, "onnxruntime.dll")) {
		t.Fatalf("cuda resolved a runtime without onnxruntime_providers_cuda.dll: %s", path)
	}
	if err == nil {
		t.Skipf("a real CUDA package elsewhere resolved first: %s", path)
	}
	if !strings.Contains(err.Error(), "onnxruntime_providers_cuda.dll") {
		t.Errorf("error does not name the missing provider: %v", err)
	}
}

func TestInKindTree(t *testing.T) {
	if !inKindTree(`C:\x\Assets\onnx\directml\Pkg\runtimes\win-x64\native`, RuntimeDirectML) {
		t.Error("directml package dir not recognised")
	}
	if inKindTree(`C:\Windows\System32`, RuntimeDirectML) {
		t.Error("System32 must not count as a directml package tree")
	}
}

func TestSearchBasesIncludesONNXRoot(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LAYA_TRT_ONNX_ROOT", root)
	want, err := filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, base := range searchBases() {
		if strings.EqualFold(base, want) {
			return
		}
	}
	t.Fatalf("searchBases() did not include LAYA_TRT_ONNX_ROOT=%q", want)
}
