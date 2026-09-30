// Command layatrt-doctor checks whether this machine can run laya-trt, and says
// exactly what to install when it cannot.
//
// Why a separate binary: the GUI and the server link against
// qualityscaler_tensorrt.dll, so Windows resolves that dependency *before*
// main() runs. When a DLL is missing they die with exit code 0xC0000279 and no
// output at all — the user gets a number, not a diagnosis.
//
// This command deliberately uses no cgo, so it always starts. It is the thing to
// run first, and the thing the launch scripts call before starting anything.
//
//	layatrt-doctor            # human-readable report
//	layatrt-doctor --json     # machine-readable, for scripts
//	layatrt-doctor --quiet    # exit code only: 0 = ready, 1 = not ready
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Check is one diagnostic result.
type Check struct {
	Name     string `json:"name"`
	OK       bool   `json:"ok"`
	Required bool   `json:"required"`
	Detail   string `json:"detail,omitempty"`
	// Fix is the concrete next action when OK is false.
	Fix string `json:"fix,omitempty"`
	// Found lists the paths that satisfied the check.
	Found []string `json:"found,omitempty"`
}

// Report is the whole diagnosis.
type Report struct {
	Ready  bool    `json:"ready"`
	Checks []Check `json:"checks"`
}

func main() {
	asJSON := flag.Bool("json", false, "print a JSON report")
	quiet := flag.Bool("quiet", false, "print nothing; exit code only")
	repoRoot := flag.String("root", "", "repository root (default: the executable's parent, else cwd)")
	flag.Parse()

	root := resolveRoot(*repoRoot)
	rep := diagnose(root)

	switch {
	case *asJSON:
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rep)
	case *quiet:
		// Exit code only, and no output at all: a PowerShell caller reads
		// $LASTEXITCODE, and any stray text would be captured as a value.
	default:
		printReport(rep, root)
	}

	if rep.Ready {
		os.Exit(0)
	}
	os.Exit(1)
}

// resolveRoot finds the repository root: an explicit flag, then the directory
// holding the built binaries, then the working directory.
func resolveRoot(explicit string) string {
	if explicit != "" {
		if abs, err := filepath.Abs(explicit); err == nil {
			return abs
		}
	}
	if exe, err := os.Executable(); err == nil {
		// A binary built into the repo sits a couple of levels down.
		dir := filepath.Dir(exe)
		for i := 0; i < 4; i++ {
			if looksLikeRepo(dir) {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		dir := cwd
		for i := 0; i < 4; i++ {
			if looksLikeRepo(dir) {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
		return cwd
	}
	return "."
}

func looksLikeRepo(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "go.mod")); err != nil {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, "include", "ai_tensorrt_cpp.h"))
	return err == nil
}

// diagnose runs every check. Required checks gate readiness; optional ones only
// matter for development.
func diagnose(root string) Report {
	checks := []Check{}

	checks = append(checks, checkKernel(root))
	checks = append(checks, checkTensorRT())
	checks = append(checks, checkCUDA())
	checks = append(checks, checkVisualCRuntime())
	checks = append(checks, checkTokenizer())
	checks = append(checks, checkEngines(root))
	checks = append(checks, checkGPU())

	// The ONNX kernel is optional: the TensorRT path alone is a complete
	// installation, so a missing ONNX Runtime is reported and never blocks.
	// Both checks stay pure file inspection, because this command must start on
	// a machine where the TensorRT DLL cannot be resolved — that is the whole
	// reason it exists.
	checks = append(checks, checkONNXBridge(root))
	checks = append(checks, checkONNXRuntime(root))

	// Development-only: reported but never blocking, because a user who just
	// wants to run the GUI does not need Go or Node.
	checks = append(checks, checkGo())
	checks = append(checks, checkCCompiler())
	checks = append(checks, checkTool("node", "Node.js", false,
		"只影响构建前端；运行已构建的程序不需要"))
	checks = append(checks, checkTool("wails", "Wails CLI", false,
		"只影响构建桌面端：go install github.com/wailsapp/wails/v2/cmd/wails@latest"))
	checks = append(checks, checkTool("trtexec", "trtexec", false,
		"只影响编译 engine；已有 engine 时不需要"))

	ready := true
	for _, c := range checks {
		if c.Required && !c.OK {
			ready = false
		}
	}
	return Report{Ready: ready, Checks: checks}
}

// ── individual checks ───────────────────────────────────────────────────────

// checkONNXBridge looks for the native ONNX bridge library.
//
// Without it the ONNX kernel is unavailable, but the TensorRT path is unaffected,
// so this is never a blocking check.
func checkONNXBridge(root string) Check {
	c := Check{
		Name:     "ONNX 内核桥 (layatrt_onnx.dll)",
		Required: false,
	}
	candidates := []string{
		filepath.Join(root, "build", "bin", "Release", "layatrt_onnx.dll"),
		filepath.Join(root, "build", "bin", "layatrt_onnx.dll"),
	}
	if runtime.GOOS != "windows" {
		candidates = []string{
			filepath.Join(root, "build", "lib", "liblayatrt_onnx.so"),
		}
	}
	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			c.OK = true
			c.Found = append(c.Found, p)
		}
	}
	if !c.OK {
		c.Detail = "ONNX 内核还没编译；只用 TensorRT 的话不影响"
		c.Fix = "运行 .\\build.ps1（需要 third_party/onnxruntime/onnxruntime_c_api.h）"
	}
	return c
}

// checkONNXRuntime looks for onnxruntime.dll, which the bridge loads at run time.
//
// It searches the places the bridge itself searches — beside the executable,
// then the repository's build output — so a pass here means the ONNX kernel will
// actually start.
func checkONNXRuntime(root string) Check {
	c := Check{
		Name:     "ONNX Runtime (onnxruntime.dll)",
		Required: false,
	}
	if runtime.GOOS != "windows" {
		c.Name = "ONNX Runtime (libonnxruntime.so)"
	}

	names := []string{"onnxruntime.dll"}
	if runtime.GOOS != "windows" {
		names = []string{"libonnxruntime.so", "onnxruntime.so"}
	}

	var candidates []string
	for _, name := range names {
		candidates = append(candidates,
			filepath.Join(root, "build", "bin", "Release", name),
			filepath.Join(root, "build", "bin", name),
			filepath.Join(root, name),
			filepath.Join(root, "runtime", name),
			filepath.Join(root, "Assets", name),
		)
	}
	// The environment override the runtime honours, so a user who has set it
	// does not get a false negative.
	if v := os.Getenv("LAYA_TRT_ONNX_RUNTIME"); v != "" {
		candidates = append([]string{v}, candidates...)
	}

	for _, p := range candidates {
		if p == "" {
			continue
		}
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			c.OK = true
			c.Found = append(c.Found, p)
		}
	}
	if c.OK {
		return c
	}

	// A copy on PATH is just as usable, so check there before reporting a miss.
	if p, err := exec.LookPath(names[0]); err == nil {
		c.OK = true
		c.Found = append(c.Found, p)
		return c
	}

	c.Detail = "没有找到 ONNX Runtime；-backend onnx 会失败，TensorRT 不受影响"
	c.Fix = "从 https://github.com/microsoft/onnxruntime/releases 取 onnxruntime.dll，" +
		"放到程序旁边或仓库根目录，或设 LAYA_TRT_ONNX_RUNTIME 指向它"
	return c
}

func checkKernel(root string) Check {
	c := Check{
		Name:     "内核 DLL (qualityscaler_tensorrt.dll)",
		Required: true,
	}
	candidates := []string{
		filepath.Join(root, "build", "bin", "Release", "qualityscaler_tensorrt.dll"),
		filepath.Join(root, "build", "bin", "qualityscaler_tensorrt.dll"),
		filepath.Join(root, "build", "Release", "qualityscaler_tensorrt.dll"),
	}
	if runtime.GOOS != "windows" {
		candidates = []string{
			filepath.Join(root, "build", "lib", "libqualityscaler_tensorrt.so"),
			filepath.Join(root, "build", "bin", "libqualityscaler_tensorrt.so"),
		}
	}
	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			c.OK = true
			c.Found = append(c.Found, p)
		}
	}
	if !c.OK {
		c.Detail = "内核还没编译，或者编译产物被删掉了"
		c.Fix = "运行 .\\build.ps1 编译内核（需要 Visual Studio 2022 + CMake）"
	}
	return c
}

// tensorRTRoots lists the places a TensorRT install is looked for.
func tensorRTRoots() []string {
	var roots []string
	for _, env := range []string{"TENSORRT_ROOT", "TensorRT_ROOT", "TENSORRT_DIR"} {
		if v := os.Getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	// Conventional install locations, newest-looking first.
	if matches, err := filepath.Glob(`C:\TensorRT-*`); err == nil {
		sort.Sort(sort.Reverse(sort.StringSlice(matches)))
		roots = append(roots, matches...)
	}
	roots = append(roots, `C:\TensorRT`)
	return roots
}

func cudaRoots() []string {
	var roots []string
	for _, env := range []string{"CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"} {
		if v := os.Getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	if matches, err := filepath.Glob(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`); err == nil {
		sort.Sort(sort.Reverse(sort.StringSlice(matches)))
		roots = append(roots, matches...)
	}
	roots = append(roots, `C:\CUDA`)
	return roots
}

func checkTensorRT() Check {
	c := Check{Name: "TensorRT 10.x (nvinfer_10.dll)", Required: true}
	if runtime.GOOS != "windows" {
		c.Name = "TensorRT 10.x (libnvinfer.so.10)"
	}
	for _, root := range tensorRTRoots() {
		for _, sub := range []string{"bin", "lib", ""} {
			dir := root
			if sub != "" {
				dir = filepath.Join(root, sub)
			}
			names := []string{"nvinfer_10.dll"}
			if runtime.GOOS != "windows" {
				names = []string{"libnvinfer.so.10", "libnvinfer.so"}
			}
			for _, name := range names {
				p := filepath.Join(dir, name)
				if _, err := os.Stat(p); err == nil {
					c.OK = true
					c.Found = append(c.Found, p)
				}
			}
		}
	}
	if !c.OK {
		c.Detail = "没找到 TensorRT 10.x 的运行库"
		c.Fix = "安装 TensorRT 10.x，然后设置 TENSORRT_ROOT 指向安装目录：" +
			"$env:TENSORRT_ROOT = 'C:\\TensorRT-10.16.0.72'；下载见 https://developer.nvidia.com/tensorrt"
	}
	return c
}

func checkCUDA() Check {
	c := Check{Name: "CUDA 运行时 (cudart64_*.dll)", Required: true}
	var all []string
	for _, root := range cudaRoots() {
		for _, sub := range []string{"bin\\x64", "bin", "lib\\x64", "lib", ""} {
			dir := root
			if sub != "" {
				dir = filepath.Join(root, sub)
			}
			patterns := []string{"cudart64_*.dll"}
			if runtime.GOOS != "windows" {
				patterns = []string{"libcudart.so*"}
			}
			for _, pat := range patterns {
				if matches, err := filepath.Glob(filepath.Join(dir, pat)); err == nil {
					all = append(all, matches...)
				}
			}
		}
	}
	c.Found = dedupe(all)
	c.OK = len(c.Found) > 0

	if !c.OK {
		c.Detail = "没找到 CUDA 运行时"
		c.Fix = "安装 CUDA Toolkit 12.x 或 13.x，或设置 CUDA_ROOT；" +
			"下载见 https://developer.nvidia.com/cuda-downloads"
		return c
	}

	// More than one CUDA on the machine is a real failure mode: the kernel was
	// linked against one of them, and whichever appears first on PATH wins. That
	// mismatch shows up as an unexplained crash, so say it here.
	if roots := distinctParents(c.Found); len(roots) > 1 {
		c.Detail = fmt.Sprintf("发现 %d 份 CUDA 运行时；内核是针对其中一份链接的，"+
			"PATH 上先出现的会被加载，选错会崩溃。env.ps1 只保留一份。", len(roots))
		c.Found = roots
	}
	return c
}

// distinctParents collapses file paths to their install root, so "several
// cudart DLLs from one CUDA" does not look like several CUDA installs.
//
// The toolkit layout is <root>\bin\x64, while a plain CUDA copy is <root>\bin;
// both are stepped up to <root>.
func distinctParents(paths []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range paths {
		dir := filepath.Dir(p)
		switch strings.ToLower(filepath.Base(dir)) {
		case "x64":
			// <root>\bin\x64 -> <root>
			dir = filepath.Dir(filepath.Dir(dir))
		case "bin", "lib":
			// <root>\bin -> <root>
			dir = filepath.Dir(dir)
		}
		if !seen[dir] {
			seen[dir] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// checkVisualCRuntime looks for the MSVC redistributable, which the kernel DLL
// needs. It is usually present, but a bare Windows install can lack it.
func checkVisualCRuntime() Check {
	c := Check{Name: "MSVC 运行时 (vcruntime140.dll)", Required: true}
	if runtime.GOOS != "windows" {
		c.OK = true
		c.Detail = "Windows 专有，跳过"
		return c
	}
	system32 := filepath.Join(os.Getenv("SystemRoot"), "System32")
	for _, name := range []string{"vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"} {
		p := filepath.Join(system32, name)
		if _, err := os.Stat(p); err == nil {
			c.Found = append(c.Found, p)
		}
	}
	c.OK = len(c.Found) > 0
	if !c.OK {
		c.Detail = "缺少 Visual C++ 运行库，内核 DLL 无法加载"
		c.Fix = "安装 Microsoft Visual C++ Redistributable (x64)：" +
			"https://aka.ms/vs/17/release/vc_redist.x64.exe"
	}
	return c
}

func checkTokenizer() Check {
	c := Check{Name: "tokenizer.json", Required: true}
	candidates := []string{}
	if v := os.Getenv("LAYA_TRT_TOKENIZER"); v != "" {
		candidates = append(candidates, v)
	}
	if home, err := os.UserHomeDir(); err == nil {
		if matches, _ := filepath.Glob(filepath.Join(home, ".cache", "huggingface", "hub",
			"models--convaiinnovations--laya", "snapshots", "*", "tokenizer", "tokenizer.json")); len(matches) > 0 {
			sort.Strings(matches)
			candidates = append(candidates, matches[len(matches)-1])
		}
	}
	candidates = append(candidates,
		filepath.Join("tokenizer", "tokenizer.json"),
		filepath.Join("models", "laya", "tokenizer", "tokenizer.json"))

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			c.OK = true
			c.Found = append(c.Found, p)
		}
	}
	if !c.OK {
		c.Detail = "没找到分词器，程序无法把文本转成 token"
		c.Fix = "下载 checkpoint：" +
			"huggingface-cli download convaiinnovations/laya --include \"tokenizer/*\" \"rl_agent_config.json\""
	}
	return c
}

func checkEngines(root string) Check {
	c := Check{Name: "TensorRT engine (*.engine)", Required: true}
	dirs := []string{
		filepath.Join(root, "AI-tensorrt"),
		filepath.Join(root, "models"),
	}
	if v := os.Getenv("LAYA_TRT_ENGINE"); v != "" {
		if _, err := os.Stat(v); err == nil {
			c.OK = true
			c.Found = append(c.Found, v)
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs,
			filepath.Join(home, "Documents", "laya-ort-probe"),
			filepath.Join(home, "Documents", "laya-trt"),
		)
	}
	dirs = append(dirs, ".")

	for _, dir := range dirs {
		matches, err := filepath.Glob(filepath.Join(dir, "*.engine"))
		if err != nil {
			continue
		}
		for _, m := range matches {
			if info, err := os.Stat(m); err == nil && !info.IsDir() && info.Size() > 0 {
				c.Found = append(c.Found, m)
			}
		}
	}
	c.OK = len(c.Found) > 0
	if !c.OK {
		c.Detail = "没有任何 engine，程序没有模型可跑"
		c.Fix = "先导出 ONNX 再编译 engine，两步都有脚本：\n" +
			"      python tools\\export_onnx.py --out laya_dyn.onnx\n" +
			"      trtexec --onnx=laya_dyn.onnx --saveEngine=laya_1k.engine --fp16 ^\n" +
			"        --minShapes=input_ids:1x64,attention_mask:1x64,marker_pos:1x2,marker_mask:1x2,qtype:1 ^\n" +
			"        --optShapes=input_ids:1x256,attention_mask:1x256,marker_pos:1x8,marker_mask:1x8,qtype:1 ^\n" +
			"        --maxShapes=input_ids:1x1024,attention_mask:1x1024,marker_pos:1x32,marker_mask:1x32,qtype:1"
	}
	return c
}

func checkGPU() Check {
	c := Check{Name: "NVIDIA GPU", Required: false}
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		c.Detail = "没找到 nvidia-smi；如果是非 NVIDIA 机器，本程序无法运行"
		c.Fix = "需要一张支持 TensorRT 10 的 NVIDIA 显卡"
		return c
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
		"--format=csv,noheader").Output()
	if err != nil {
		c.Detail = "nvidia-smi 执行失败"
		c.Fix = "检查显卡驱动是否正常"
		return c
	}
	line := strings.TrimSpace(string(out))
	if line == "" {
		c.Detail = "nvidia-smi 没有报告任何显卡"
		c.Fix = "检查显卡驱动是否正常"
		return c
	}
	c.OK = true
	c.Found = []string{strings.Split(line, "\n")[0]}
	return c
}

func checkTool(name, label string, required bool, fix string) Check {
	c := Check{Name: label, Required: required}
	path, err := exec.LookPath(name)
	if err != nil {
		c.Detail = "不在 PATH 上"
		c.Fix = fix
		return c
	}
	c.OK = true
	c.Found = []string{path}
	return c
}

// checkGo reports the Go toolchain and, importantly, whether CGO is enabled.
//
// This project's kernel package is cgo, so a build with CGO_ENABLED=0 fails with
// "build constraints exclude all Go files" — which says nothing about cgo being
// the cause. Surfacing it here saves that detour.
func checkGo() Check {
	c := Check{Name: "Go 工具链 (含 CGO)", Required: false}
	path, err := exec.LookPath("go")
	if err != nil {
		c.Detail = "不在 PATH 上"
		c.Fix = "只影响从源码构建；运行已构建的程序不需要。" +
			"装 Go 1.25+：https://go.dev/dl/"
		return c
	}
	c.Found = []string{path}

	out, err := exec.Command("go", "env", "CGO_ENABLED").Output()
	if err != nil {
		c.Detail = "无法读取 go env CGO_ENABLED"
		return c
	}
	if strings.TrimSpace(string(out)) != "1" {
		c.OK = true // go itself is present; this is a caveat, not a failure
		c.Detail = "CGO_ENABLED 是 0，直接 go build 会失败并报 " +
			"\"build constraints exclude all Go files\"（内核是 cgo 包）"
		c.Fix = "$env:CGO_ENABLED = '1'   # 编译前必须设置"
		return c
	}
	c.OK = true
	return c
}

// checkCCompiler looks for the C compiler cgo will actually invoke.
//
// Go's cgo driver uses CC (default "gcc"). Having Visual Studio installed is not
// enough: cl.exe has to be on PATH, which normally means running vcvars64 first.
// So this checks the tool cgo will call, and mentions MSVC only as a route to
// getting a working CC.
func checkCCompiler() Check {
	c := Check{Name: "C 编译器 (CGO 依赖)", Required: false}

	cc := "gcc"
	if out, err := exec.Command("go", "env", "CC").Output(); err == nil {
		if v := strings.TrimSpace(string(out)); v != "" {
			cc = v
		}
	}
	path, err := exec.LookPath(cc)
	if err == nil {
		c.OK = true
		c.Found = []string{path}
		return c
	}

	c.Detail = fmt.Sprintf("cgo 要用的 C 编译器 %q 不在 PATH 上，go build 会失败并报 "+
		"\"cgo: C compiler \\\"gcc\\\" not found\"", cc)
	if hasMSVC() {
		c.Fix = "机器上有 Visual Studio，但 cl.exe 默认不在 PATH 上。" +
			"编译前先跑 vcvars64：\n" +
			"      call \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat\"\n" +
			"      或装 mingw-w64 并把它的 bin 加进 PATH（更省事）"
	} else {
		c.Fix = "装 mingw-w64（推荐，直接提供 gcc）或 Visual Studio 2022 的 C++ 工作负载；" +
			"只影响从源码构建"
	}
	return c
}

// hasMSVC reports whether vswhere can find a Visual Studio C++ toolchain.
func hasMSVC() bool {
	vswhere := filepath.Join(os.Getenv("ProgramFiles(x86)"),
		"Microsoft Visual Studio", "Installer", "vswhere.exe")
	if _, err := os.Stat(vswhere); err != nil {
		return false
	}
	out, err := exec.Command(vswhere, "-latest", "-products", "*",
		"-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
		"-property", "installationPath").Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
}

// ── reporting ───────────────────────────────────────────────────────────────

func printReport(rep Report, root string) {
	fmt.Printf("Laya Go Launcher 环境自检\n")
	fmt.Printf("仓库: %s\n", root)
	fmt.Printf("系统: %s/%s\n\n", runtime.GOOS, runtime.GOARCH)

	for _, c := range rep.Checks {
		mark := "✓"
		switch {
		case c.OK:
			mark = "\u2713"
		case c.Required:
			mark = "\u2717"
		default:
			mark = "·"
		}
		line := fmt.Sprintf("  %s %s", mark, c.Name)
		if len(c.Found) > 0 && c.OK {
			line += "  " + firstLine(c.Found[0])
			if len(c.Found) > 1 {
				line += fmt.Sprintf("  (+%d)", len(c.Found)-1)
			}
		}
		fmt.Println(line)

		// A detail on a passing check is a caveat, not a failure; show it, along
		// with any remedy it carries.
		if c.OK && (c.Detail != "" || c.Fix != "") {
			if c.Detail != "" {
				fmt.Printf("      ! %s\n", c.Detail)
			}
			for _, f := range c.Found[min(1, len(c.Found)):] {
				fmt.Printf("        %s\n", f)
			}
			if c.Fix != "" {
				for _, l := range strings.Split(c.Fix, "\n") {
					fmt.Printf("      → %s\n", l)
				}
			}
		}

		if !c.OK {
			if c.Detail != "" {
				fmt.Printf("      %s\n", c.Detail)
			}
			if c.Fix != "" {
				for _, l := range strings.Split(c.Fix, "\n") {
					fmt.Printf("      → %s\n", l)
				}
			}
		}
	}

	fmt.Println()
	if rep.Ready {
		fmt.Println("就绪。启动：")
		fmt.Println("  .\\run.ps1            启动桌面端")
		fmt.Println("  .\\run.ps1 -Server    只启动 HTTP API")
		return
	}

	fmt.Println("还不能运行。按上面的 → 逐条处理后重试。")
	fmt.Println("想一次装齐构建环境，运行：.\\setup.ps1")
	os.Exit(1)
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
