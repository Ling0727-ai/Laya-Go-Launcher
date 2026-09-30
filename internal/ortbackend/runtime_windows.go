//go:build windows

package ortbackend

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/local/laya-go-launcher/internal/backend"
)

// ONNX Runtime discovery.
//
// This is the laya-trt port of QualityScaler-go's findRuntimeDLL / onnxRuntime-
// PackageCandidates. The problem it solves is that "onnxruntime.dll" is not one
// library but a family: a CPU build, a CUDA build, and a DirectML build ship the
// same file name and are not interchangeable, and each has provider DLLs that
// must come from the same package. A machine commonly has several of them — the
// Python wheel, a vendored copy under a sibling project, one on PATH — and
// picking the wrong one produces a session that either cannot use the GPU or
// cannot load at all.
//
// QS's answer, kept here, is to search a ranked list of locations and to prefer a
// runtime whose matching provider library sits beside it. What is deliberately
// NOT ported is QS's process-wide state (a single resolved path cached in a
// package variable, plus a mutated PATH): this program can hold a TensorRT model
// and an ONNX session at the same time, and it must not edit the process PATH as
// a side effect of loading a model.

// RuntimeKind names one distribution of ONNX Runtime. The values are the
// directory names QS uses under Assets/onnx, so an existing QS-style layout is
// recognised as-is.
type RuntimeKind string

const (
	RuntimeCUDA13   RuntimeKind = "cuda13"
	RuntimeCUDA12   RuntimeKind = "cuda12"
	RuntimeDirectML RuntimeKind = "directml"
	RuntimeCPU      RuntimeKind = "cpu"
	RuntimeAny      RuntimeKind = "any"
)

// runtimeSpec describes how one distribution is laid out on disk.
type runtimeSpec struct {
	kind RuntimeKind
	// main is the runtime library's file name.
	main string
	// provider is the provider library that must accompany it, when the
	// distribution has one. An empty value means the runtime is self-contained.
	provider string
	// providerGlob matches the provider library inside a package tree, where the
	// version is part of the directory name.
	providerGlob string
}

func windowsRuntimeSpecs() []runtimeSpec {
	return []runtimeSpec{
		{kind: RuntimeCUDA13, main: "onnxruntime.dll", provider: "onnxruntime_providers_cuda.dll",
			providerGlob: "onnxruntime_providers_cuda.dll"},
		{kind: RuntimeCUDA12, main: "onnxruntime.dll", provider: "onnxruntime_providers_cuda.dll",
			providerGlob: "onnxruntime_providers_cuda.dll"},
		{kind: RuntimeDirectML, main: "onnxruntime.dll", provider: "DirectML.dll",
			providerGlob: "DirectML.dll"},
		{kind: RuntimeCPU, main: "onnxruntime.dll"},
	}
}

// specsForProvider maps a requested execution provider onto the runtime
// distributions that can serve it, most specific first.
//
// This is the check that prevents the classic mismatch: asking for CUDA while
// resolving a DirectML runtime makes ORT look for onnxruntime_providers_cuda.dll
// under the DirectML directory, where it will never be. Naming the provider
// therefore constrains which runtime is acceptable, rather than being applied
// after the fact.
func specsForProvider(provider string) []runtimeSpec {
	all := windowsRuntimeSpecs()
	want := strings.ToLower(strings.TrimSpace(provider))
	switch want {
	case "cuda":
		return filterSpecs(all, RuntimeCUDA13, RuntimeCUDA12)
	case "directml", "dml":
		return filterSpecs(all, RuntimeDirectML)
	case "cpu":
		// CPU is the fallback every distribution can serve, but a CPU-only build
		// is the one that cannot accidentally pull in a GPU provider.
		return filterSpecs(all, RuntimeCPU)
	default:
		// Unspecified: any distribution is acceptable. CUDA first because this
		// program targets an NVIDIA machine by default, then DirectML for the
		// AMD/Intel case, then the CPU build.
		return filterSpecs(all, RuntimeCUDA13, RuntimeCUDA12, RuntimeDirectML, RuntimeCPU)
	}
}

func filterSpecs(all []runtimeSpec, kinds ...RuntimeKind) []runtimeSpec {
	want := make(map[RuntimeKind]bool, len(kinds))
	for _, k := range kinds {
		want[k] = true
	}
	var out []runtimeSpec
	for _, s := range all {
		if want[s.kind] {
			out = append(out, s)
		}
	}
	return out
}

// searchBases are the roots a package tree is looked under, most specific first.
//
// The executable's own directory comes first so a deployed app uses the runtime
// shipped beside it, ahead of anything the developer machine happens to have.
// The parent directories cover the "binary in build/bin/Release, assets at the
// repository root" layout this project uses.
func searchBases() []string {
	var bases []string
	add := func(p string) {
		if p == "" {
			return
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			return
		}
		for _, existing := range bases {
			if strings.EqualFold(existing, abs) {
				return
			}
		}
		bases = append(bases, abs)
	}

	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		add(dir)
		// Walk up a few levels: a binary built into build/bin/Release sits three
		// directories below the repository root that holds Assets/.
		up := dir
		for i := 0; i < 4; i++ {
			parent := filepath.Dir(up)
			if parent == up {
				break
			}
			up = parent
			add(up)
		}
	}
	if runtimeRoot := os.Getenv("LAYA_TRT_ONNX_ROOT"); runtimeRoot != "" {
		add(runtimeRoot)
	}
	// Reuse a sibling QualityScaler-go install's Assets/onnx tree, so the GUI
	// finds the CUDA/DirectML packages even when launched without env.ps1.
	// Without it discovery lands on the inbox System32 CPU-only runtime.
	var siblings []string
	if home, err := os.UserHomeDir(); err == nil {
		for _, qs := range []string{
			filepath.Join(home, "Downloads", "QualityScaler-go"),
			filepath.Join(home, "Documents", "QualityScaler-go"),
			filepath.Join(home, "Desktop", "QualityScaler-go"),
		} {
			if info, err := os.Stat(filepath.Join(qs, "Assets", "onnx")); err == nil && info.IsDir() {
				siblings = append(siblings, qs)
			}
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		add(cwd)
		up := cwd
		for i := 0; i < 3; i++ {
			parent := filepath.Dir(up)
			if parent == up {
				break
			}
			up = parent
			add(up)
		}
	}
	// Last, so a local Assets tree or LAYA_TRT_ONNX_ROOT still wins.
	for _, qs := range siblings {
		add(qs)
	}
	return bases
}

// runtimeCandidates returns the paths to try for one distribution, in order.
//
// The shapes covered are the ones that actually occur:
//
//   - <base>/onnxruntime.dll                      — deployed beside the binary
//   - <base>/runtime/onnxruntime.dll              — this project's convention
//   - <base>/Assets/onnxruntime.dll               — QualityScaler-go's convention
//   - <base>/Assets/onnx/<kind>/onnxruntime.dll   — QS's versioned layout
//   - <base>/Assets/onnx/<kind>/*/lib/onnxruntime.dll
//   - <base>/Assets/onnx/<kind>/*/runtimes/win-x64/native/onnxruntime.dll
//
// The globbed forms matter because QS ships packages whose directory carries the
// version, so the exact path cannot be hardcoded.
func runtimeCandidates(base string, spec runtimeSpec) []string {
	var out []string
	direct := []string{
		filepath.Join(base, spec.main),
		filepath.Join(base, "runtime", spec.main),
		filepath.Join(base, "Assets", spec.main),
		filepath.Join(base, "res", spec.main),
		filepath.Join(base, "lib", spec.main),
	}
	if spec.kind == RuntimeCPU {
		out = append(out, direct...)
	}

	// Prefer a provider-specific package over a generic Assets/onnxruntime.dll.
	// The latter may have a CUDA provider DLL beside it but depend on a different
	// cuDNN release, so selecting it first breaks an otherwise valid cuda13 setup.
	for _, kindDir := range kindDirNames(spec.kind) {
		root := filepath.Join(base, "Assets", "onnx", kindDir)
		out = append(out,
			filepath.Join(root, spec.main),
			filepath.Join(root, "lib", spec.main),
			filepath.Join(root, "*", spec.main),
			filepath.Join(root, "*", "lib", spec.main),
			filepath.Join(root, "runtimes", "win-x64", "native", spec.main),
			filepath.Join(root, "*", "runtimes", "win-x64", "native", spec.main),
			filepath.Join(root, "*", "lib", "runtimes", "win-x64", "native", spec.main),
		)
	}
	if spec.kind != RuntimeCPU {
		out = append(out, direct...)
	}
	return out
}

// kindDirNames lists the directory names a distribution may live under. The
// QS tree uses the bare kind ("cuda13"); the sibling "AI-onnx" tree this project
// inherited uses it too, so both are covered by the same list.
func kindDirNames(kind RuntimeKind) []string {
	switch kind {
	case RuntimeCUDA13:
		return []string{"cuda13", "cuda"}
	case RuntimeCUDA12:
		return []string{"cuda12", "cuda"}
	case RuntimeDirectML:
		return []string{"directml", "dml"}
	case RuntimeCPU:
		return []string{"cpu"}
	default:
		return []string{"cuda13", "cuda12", "directml", "cpu"}
	}
}

// resolveRuntime finds the ONNX Runtime library to load.
//
// explicit wins outright: a user who names a path gets that path, and a failure
// to load it is reported rather than quietly substituted, for the same reason an
// explicit backend name is never substituted.
//
// Otherwise every acceptable distribution is tried in preference order. A
// candidate whose provider library sits beside it wins immediately; one whose
// provider is missing is remembered but not returned while a matched candidate
// might still be found. That ordering is what keeps a CUDA request from resolving
// a DirectML build, because the file name alone cannot tell those apart.
//
// If no candidate is provider-matched but a runtime does exist, the first one is
// returned anyway. That is deliberate: the bridge reports which execution
// provider it actually got and why the requested one was skipped, so a model that
// runs on the CPU is more useful than a load that refuses — the same trade the
// provider selection itself already makes. A hard error is reserved for the case
// where no runtime library exists at all, which is not a fallback but a missing
// dependency.
//
// Returns the resolved path and the distribution it came from.
func resolveRuntime(explicit, provider string) (string, RuntimeKind, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", RuntimeAny, fmt.Errorf("%w: %v", backend.ErrRuntimeMissing, err)
		}
		abs, err := filepath.Abs(explicit)
		if err != nil {
			abs = explicit
		}
		return abs, RuntimeAny, nil
	}

	specs := specsForProvider(provider)
	bases := searchBases()

	var (
		fallbackPath string
		fallbackKind RuntimeKind
		unmatched    int
	)
	for _, spec := range specs {
		for _, base := range bases {
			for _, candidate := range runtimeCandidates(base, spec) {
				matches := []string{candidate}
				if strings.ContainsAny(candidate, "*?") {
					found, err := filepath.Glob(candidate)
					if err != nil {
						continue
					}
					// A glob can match several versions; take the last after
					// sorting, which is the newest-looking one.
					sort.Strings(found)
					matches = found
				}
				for _, match := range matches {
					info, err := os.Stat(match)
					if err != nil || info.IsDir() {
						continue
					}
					abs, err := filepath.Abs(match)
					if err != nil {
						abs = match
					}
					if providerPresent(spec, filepath.Dir(abs)) {
						return abs, spec.kind, nil
					}
					unmatched++
					if fallbackPath == "" {
						fallbackPath, fallbackKind = abs, spec.kind
					}
				}
			}
		}
	}

	// An explicit GPU provider is strict: a runtime without that provider's
	// library (a CPU build, or the inbox System32 onnxruntime.dll) can only
	// produce a CPU session, which the load then rejects with a confusing
	// "provider not present in this build". Say where the package was expected.
	if want := strings.ToLower(strings.TrimSpace(provider)); want == "cuda" || want == "directml" || want == "dml" {
		dirs := make([]string, 0, 2)
		for _, spec := range specs {
			dirs = append(dirs, filepath.Join("Assets", "onnx", kindDirNames(spec.kind)[0]))
		}
		detail := ""
		if fallbackPath != "" {
			detail = fmt.Sprintf(" (found %s, but %s is not beside it)", fallbackPath, specs[0].provider)
		}
		return "", RuntimeAny, fmt.Errorf(
			"%w: no %s ONNX Runtime package found%s; put it under %s in one of: %s, or set LAYA_TRT_ONNX_ROOT",
			backend.ErrRuntimeMissing, want, detail, strings.Join(dirs, " / "), strings.Join(bases, ", "))
	}

	if fallbackPath != "" {
		// A runtime exists but its own provider library was not beside it. Return
		// it and let the bridge report the provider outcome, rather than refusing
		// a load that would otherwise work.
		return fallbackPath, fallbackKind, nil
	}

	// Nothing was found in the search bases, so fall back to the DLL search path
	// itself. This is the case the bridge's own discovery handled before, and it
	// covers two real layouts: a runtime installed into System32 (Windows ships
	// onnxruntime.dll there for the inbox ML stack), and one a user put on PATH.
	//
	// The distribution cannot be identified from a bare name, so it is reported
	// as any: the bridge still names the provider it actually got, which is where
	// that detail belongs.
	if p, err := exec.LookPath("onnxruntime.dll"); err == nil {
		if abs, absErr := filepath.Abs(p); absErr == nil {
			return abs, RuntimeAny, nil
		}
		return p, RuntimeAny, nil
	}

	// Report the bases searched, because "onnxruntime.dll not found" without
	// saying where was looked is not actionable.
	return "", RuntimeAny, fmt.Errorf(
		"%w: no onnxruntime.dll found for provider %q; searched under %s and on PATH",
		backend.ErrRuntimeMissing, providerOrAny(provider), strings.Join(bases, ", "))
}

// providerPresent reports whether a distribution's provider library accompanies
// the runtime.
//
// It checks the runtime's own directory and the immediate subdirectories, which
// is where every packaged layout puts them: a nuget package keeps both under
// runtimes/win-x64/native, and the QualityScaler-go tree keeps both in the
// package's lib. A self-contained distribution (CPU) has no provider requirement
// and always passes.
//
// Deliberately no parent-directory search: accepting a provider from a sibling
// package is exactly the mismatch this check exists to prevent, and a false
// negative costs only a reported provider fallback.
func providerPresent(spec runtimeSpec, dir string) bool {
	if spec.provider == "" {
		return true
	}
	if _, err := os.Stat(filepath.Join(dir, spec.provider)); err == nil {
		return true
	}
	for _, sub := range []string{"lib", "native", filepath.Join("runtimes", "win-x64", "native")} {
		if _, err := os.Stat(filepath.Join(dir, sub, spec.provider)); err == nil {
			return true
		}
	}
	// The DirectML EP is compiled into a DirectML onnxruntime.dll; DirectML.dll
	// itself ships with Windows 10+ in System32. A nuget package that omits it
	// is still a valid DirectML distribution when it sits in a directml tree.
	if spec.kind == RuntimeDirectML && inKindTree(dir, spec.kind) {
		if sys := os.Getenv("SystemRoot"); sys != "" {
			if _, err := os.Stat(filepath.Join(sys, "System32", spec.provider)); err == nil {
				return true
			}
		}
	}
	return false
}

// inKindTree reports whether dir lies under an Assets/onnx/<kind> package tree.
func inKindTree(dir string, kind RuntimeKind) bool {
	lower := strings.ToLower(filepath.ToSlash(dir))
	for _, name := range kindDirNames(kind) {
		if strings.Contains(lower, "/assets/onnx/"+name+"/") || strings.HasSuffix(lower, "/assets/onnx/"+name) {
			return true
		}
	}
	return false
}

func providerOrAny(provider string) string {
	if strings.TrimSpace(provider) == "" {
		return "any"
	}
	return provider
}
