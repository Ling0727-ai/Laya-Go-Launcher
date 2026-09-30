package ortbackend

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// Scratch adversarial probe: prove the os.Stat in Load fires BEFORE
// newNativeSession, by giving both a missing model AND a bogus bridge/runtime.
// If newNativeSession ran first, the error would be ErrRuntimeMissing.
func TestScratchStatBeforeNativeSession(t *testing.T) {
	dir := t.TempDir()
	missingModel := filepath.Join(dir, "nope.onnx")
	bogusDLL := filepath.Join(dir, "no-such-bridge.dll")
	bogusRuntime := filepath.Join(dir, "no-such-runtime.dll")

	be := New(bogusDLL)
	defer be.Close()
	_, err := be.Load(context.Background(), backend.Options{
		Path:        missingModel,
		RuntimePath: bogusRuntime,
		Provider:    "cpu",
	})
	t.Logf("missing model + bogus bridge error = %v", err)
	t.Logf("  errors.Is(os.ErrNotExist)=%v errors.Is(ErrRuntimeMissing)=%v",
		errors.Is(err, os.ErrNotExist), errors.Is(err, backend.ErrRuntimeMissing))
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("FALSIFIED: stat did not fire first, err=%v", err)
	}
	if errors.Is(err, backend.ErrRuntimeMissing) {
		t.Errorf("newNativeSession ran before the stat: %v", err)
	}

	// Control: a file that DOES exist with a bogus bridge must report the
	// runtime problem, proving the stat is not swallowing everything.
	real := filepath.Join(dir, "present.onnx")
	if werr := os.WriteFile(real, []byte("not really a graph"), 0o600); werr != nil {
		t.Fatal(werr)
	}
	_, err2 := be.Load(context.Background(), backend.Options{
		Path:        real,
		RuntimePath: bogusRuntime,
		Provider:    "cpu",
	})
	t.Logf("present model + bogus bridge error = %v", err2)
	t.Logf("  errors.Is(os.ErrNotExist)=%v errors.Is(ErrRuntimeMissing)=%v",
		errors.Is(err2, os.ErrNotExist), errors.Is(err2, backend.ErrRuntimeMissing))
	if errors.Is(err2, os.ErrNotExist) {
		t.Errorf("an existing file was reported as not-exist: %v", err2)
	}
}
