//go:build windows

package nativeenv

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

var (
	prepareOnce sync.Once
	prepared    Dirs
)

// Prepare discovers the native runtime directories and prepends them to this
// process's PATH. It runs once; later calls return the first result.
func Prepare() Dirs {
	prepareOnce.Do(func() {
		prepared = discover()
		apply(&prepared)
	})
	return prepared
}

func discover() Dirs {
	var d Dirs
	d.Kernel = findKernel()
	d.TensorRT = findTensorRT()
	d.CUDA = findCUDA()
	d.CuDNN = findCuDNN(cudaMajor(d.CUDA))
	return d
}

// apply puts the discovered directories first on PATH, drops entries that are
// files, and removes duplicates. Relative order of the remaining entries is
// kept, so tools found through PATH (go, node, ...) are unaffected.
func apply(d *Dirs) {
	front := []string{d.Kernel, d.TensorRT}
	front = append(front, d.CUDA...)
	front = append(front, d.CuDNN)

	seen := map[string]bool{}
	var out []string
	add := func(p string) {
		if p == "" {
			return
		}
		key := strings.ToLower(strings.TrimRight(p, `\/`))
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, p)
	}
	for _, p := range front {
		add(p)
	}
	for _, p := range filepath.SplitList(os.Getenv("PATH")) {
		if p == "" {
			continue
		}
		// An entry naming a file (e.g. ...\WindowsApps\...\pwsh.exe) makes the
		// loader stop searching PATH at that point, so every directory after it
		// becomes invisible to LoadLibrary.
		if info, err := os.Stat(os.ExpandEnv(p)); err == nil && !info.IsDir() {
			d.Dropped = append(d.Dropped, p)
			continue
		}
		add(p)
	}
	_ = os.Setenv("PATH", strings.Join(out, string(os.PathListSeparator)))
}

// ── discovery ───────────────────────────────────────────────────────────────

func has(dir, name string) bool {
	if dir == "" {
		return false
	}
	info, err := os.Stat(filepath.Join(dir, name))
	return err == nil && !info.IsDir()
}

// firstWith returns the first directory that holds name.
func firstWith(name string, dirs ...string) string {
	for _, d := range dirs {
		if has(d, name) {
			if abs, err := filepath.Abs(d); err == nil {
				return abs
			}
			return d
		}
	}
	return ""
}

// ancestors returns dir and up to n of its parents.
func ancestors(dir string, n int) []string {
	out := []string{dir}
	for i := 0; i < n; i++ {
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
		out = append(out, dir)
	}
	return out
}

func pathDirs() []string {
	var out []string
	for _, p := range filepath.SplitList(os.Getenv("PATH")) {
		if p != "" {
			out = append(out, os.ExpandEnv(p))
		}
	}
	return out
}

var numRe = regexp.MustCompile(`\d+`)

// versionKey extracts the numbers in a name ("v13.3" → [13 3]).
func versionKey(name string) []int {
	var out []int
	for _, m := range numRe.FindAllString(name, -1) {
		n, _ := strconv.Atoi(m)
		out = append(out, n)
	}
	return out
}

func newerFirst(a, b string) bool {
	ka, kb := versionKey(filepath.Base(a)), versionKey(filepath.Base(b))
	for i := 0; i < len(ka) && i < len(kb); i++ {
		if ka[i] != kb[i] {
			return ka[i] > kb[i]
		}
	}
	return len(ka) > len(kb)
}

// globDirs returns matching directories, newest version first.
func globDirs(pattern string) []string {
	matches, _ := filepath.Glob(pattern)
	var out []string
	for _, m := range matches {
		if info, err := os.Stat(m); err == nil && info.IsDir() {
			out = append(out, m)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return newerFirst(out[i], out[j]) })
	return out
}

func programFiles() string {
	if p := os.Getenv("ProgramFiles"); p != "" {
		return p
	}
	return `C:\Program Files`
}

// findKernel looks beside the executable, then in build\bin\Release of the
// repository the executable or working directory lives in.
func findKernel() string {
	var dirs []string
	if v := os.Getenv("LAYA_TRT_KERNEL_DIR"); v != "" {
		dirs = append(dirs, v)
	}
	var roots []string
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, ancestors(filepath.Dir(exe), 4)...)
	}
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, ancestors(cwd, 3)...)
	}
	for _, r := range roots {
		dirs = append(dirs, r,
			filepath.Join(r, "build", "bin", "Release"),
			filepath.Join(r, "build", "Release"))
	}
	return firstWith(KernelDLL, dirs...)
}

func findTensorRT() string {
	const lib = "nvinfer_10.dll"
	var roots []string
	if v := os.Getenv("TENSORRT_ROOT"); v != "" {
		roots = append(roots, v)
	}
	roots = append(roots, globDirs(`C:\TensorRT-*`)...)
	roots = append(roots, globDirs(filepath.Join(programFiles(), "NVIDIA", "TensorRT*"))...)
	var dirs []string
	for _, r := range roots {
		dirs = append(dirs, filepath.Join(r, "bin"), filepath.Join(r, "lib"), r)
	}
	dirs = append(dirs, pathDirs()...)
	return firstWith(lib, dirs...)
}

var cudartRe = regexp.MustCompile(`(?i)^cudart64_(\d+)\.dll$`)

// cudartMajor returns the CUDA major a directory's runtime belongs to, or 0.
func cudartMajor(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	for _, e := range entries {
		if m := cudartRe.FindStringSubmatch(e.Name()); m != nil {
			n, _ := strconv.Atoi(m[1])
			// cudart64_110.dll is CUDA 11, cudart64_13.dll is CUDA 13.
			if n >= 100 {
				n /= 10
			}
			return n
		}
	}
	return 0
}

// findCUDA returns one runtime directory per CUDA major, newest major first.
// Explicit settings come first within a major, then newer toolkits.
func findCUDA() []string {
	var roots []string
	for _, env := range []string{"CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"} {
		if v := os.Getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	roots = append(roots, `C:\CUDA`)
	roots = append(roots, globDirs(filepath.Join(programFiles(), "NVIDIA GPU Computing Toolkit", "CUDA", "v*"))...)

	byMajor := map[int]string{}
	for _, r := range roots {
		for _, d := range []string{filepath.Join(r, "bin", "x64"), filepath.Join(r, "bin")} {
			major := cudartMajor(d)
			if major == 0 {
				continue
			}
			if _, ok := byMajor[major]; !ok {
				byMajor[major] = d
			}
			break
		}
	}
	majors := make([]int, 0, len(byMajor))
	for m := range byMajor {
		majors = append(majors, m)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(majors)))
	out := make([]string, 0, len(majors))
	for _, m := range majors {
		out = append(out, byMajor[m])
	}
	return out
}

func cudaMajor(dirs []string) int {
	if len(dirs) == 0 {
		return 0
	}
	return cudartMajor(dirs[0])
}

// findCuDNN picks the cuDNN build for the given CUDA major. cuDNN 9 installs
// as CUDNN\v9.x\bin\<cuda major.minor>\x64; older installs use a flat bin.
func findCuDNN(major int) string {
	const lib = "cudnn64_9.dll"
	var roots []string
	if v := os.Getenv("CUDNN_PATH"); v != "" {
		roots = append(roots, v)
	}
	roots = append(roots, globDirs(filepath.Join(programFiles(), "NVIDIA", "CUDNN", "v*"))...)
	var dirs []string
	for _, r := range roots {
		if major > 0 {
			for _, sub := range globDirs(filepath.Join(r, "bin", strconv.Itoa(major)+".*")) {
				dirs = append(dirs, filepath.Join(sub, "x64"), sub)
			}
		}
		dirs = append(dirs, filepath.Join(r, "bin", "x64"), filepath.Join(r, "bin"))
	}
	if found := firstWith(lib, dirs...); found != "" {
		return found
	}
	// Already reachable (e.g. copied into the CUDA bin): nothing to add.
	return ""
}
