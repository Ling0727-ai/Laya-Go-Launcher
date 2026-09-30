//go:build windows

package nativeenv

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func touch(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestApplyDropsFileEntries checks that a PATH entry naming a file is removed,
// discovered directories go first, and duplicates collapse.
func TestApplyDropsFileEntries(t *testing.T) {
	root := t.TempDir()
	exe := filepath.Join(root, "pwsh.exe")
	touch(t, exe)
	keep := filepath.Join(root, "tools")
	if err := os.MkdirAll(keep, 0o755); err != nil {
		t.Fatal(err)
	}
	trt := filepath.Join(root, "trt")
	t.Setenv("PATH", strings.Join([]string{keep, exe, trt, keep}, ";"))

	d := Dirs{TensorRT: trt}
	apply(&d)

	got := filepath.SplitList(os.Getenv("PATH"))
	want := []string{trt, keep}
	if strings.Join(got, ";") != strings.Join(want, ";") {
		t.Fatalf("PATH = %v, want %v", got, want)
	}
	if len(d.Dropped) != 1 || d.Dropped[0] != exe {
		t.Fatalf("Dropped = %v, want [%s]", d.Dropped, exe)
	}
}

func TestCudartMajor(t *testing.T) {
	for name, want := range map[string]int{
		"cudart64_13.dll":  13,
		"cudart64_12.dll":  12,
		"cudart64_110.dll": 11,
	} {
		dir := t.TempDir()
		touch(t, filepath.Join(dir, name))
		if got := cudartMajor(dir); got != want {
			t.Errorf("%s: major = %d, want %d", name, got, want)
		}
	}
	if got := cudartMajor(t.TempDir()); got != 0 {
		t.Errorf("empty dir: major = %d, want 0", got)
	}
}

func TestGlobDirsNewestFirst(t *testing.T) {
	root := t.TempDir()
	for _, v := range []string{"v12.9", "v13.3", "v13.10", "v9.0"} {
		if err := os.MkdirAll(filepath.Join(root, v), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got := globDirs(filepath.Join(root, "v*"))
	var names []string
	for _, g := range got {
		names = append(names, filepath.Base(g))
	}
	if strings.Join(names, ",") != "v13.10,v13.3,v12.9,v9.0" {
		t.Fatalf("order = %v", names)
	}
}

// TestFindCuDNNMatchesCUDAMajor checks the cuDNN 9 layout bin\<cuda>\x64 is
// picked for the CUDA major in use, not the first one on disk.
func TestFindCuDNNMatchesCUDAMajor(t *testing.T) {
	root := t.TempDir()
	touch(t, filepath.Join(root, "bin", "12.9", "x64", "cudnn64_9.dll"))
	touch(t, filepath.Join(root, "bin", "13.3", "x64", "cudnn64_9.dll"))
	t.Setenv("CUDNN_PATH", root)
	t.Setenv("ProgramFiles", t.TempDir()) // hide the real install

	if got, want := findCuDNN(13), filepath.Join(root, "bin", "13.3", "x64"); !strings.EqualFold(got, want) {
		t.Errorf("major 13: %s, want %s", got, want)
	}
	if got, want := findCuDNN(12), filepath.Join(root, "bin", "12.9", "x64"); !strings.EqualFold(got, want) {
		t.Errorf("major 12: %s, want %s", got, want)
	}
}

// TestFindKernelFromWorkingDirectory checks the repository layout is found
// from a subdirectory, which is where `wails dev` runs.
func TestFindKernelFromWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	touch(t, filepath.Join(root, "build", "bin", "Release", KernelDLL))
	sub := filepath.Join(root, "cmd", "layatrt-gui")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(sub)
	t.Setenv("LAYA_TRT_KERNEL_DIR", "")
	got := findKernel()
	if !strings.EqualFold(got, filepath.Join(root, "build", "bin", "Release")) {
		t.Fatalf("findKernel() = %q", got)
	}
}
