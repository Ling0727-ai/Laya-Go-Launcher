package modelstore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// touch creates an empty file so describe/scan can stat it.
func touch(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestDescribeReadsNames(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name      string
		format    Format
		seq       int
		precision string
		multi     bool
		optimized bool
	}{
		{"laya_s8192_fp16_p2.engine", FormatEngine, 8192, "fp16", true, false},
		{"laya_s1024_fp16_b8.engine", FormatEngine, 1024, "fp16", false, false},
		{"laya_ctx8192.onnx", FormatONNX, 8192, "fp32", false, false},
		{"laya_ctx8192.opt.fp16.onnx", FormatONNX, 8192, "fp16", false, true},
		{"mystery.engine", FormatEngine, 0, "", false, false},
	}
	for _, c := range cases {
		p := touch(t, dir, c.name)
		st, _ := os.Stat(p)
		m := describe(p, st)
		if m.Format != c.format || m.SeqMax != c.seq || m.Precision != c.precision ||
			m.MultiProfile != c.multi || m.Optimized != c.optimized {
			t.Errorf("%s: got format=%s seq=%d prec=%q multi=%v opt=%v", c.name,
				m.Format, m.SeqMax, m.Precision, m.MultiProfile, m.Optimized)
		}
	}
}

func repoLikeModels() []Model {
	now := time.Now()
	mk := func(name string, f Format, seq int, prec string, multi, opt bool) Model {
		return Model{Path: `C:\m\` + name, Name: name, Format: f, SeqMax: seq, Precision: prec,
			MultiProfile: multi, Optimized: opt, Modified: now, Origin: "name"}
	}
	return []Model{
		mk("laya_s512_fp16_b8.engine", FormatEngine, 512, "fp16", false, false),
		mk("laya_s8192_fp16_b1.engine", FormatEngine, 8192, "fp16", false, false),
		mk("laya_s8192_fp16_p2.engine", FormatEngine, 8192, "fp16", true, false),
		mk("mystery.engine", FormatEngine, 0, "", false, false),
		mk("laya_ctx8192.onnx", FormatONNX, 8192, "fp32", false, false),
		mk("laya_ctx8192.opt.onnx", FormatONNX, 8192, "fp32", false, true),
		mk("laya_ctx8192.opt.fp16.onnx", FormatONNX, 8192, "fp16", false, true),
	}
}

func TestPlanDefaultChain(t *testing.T) {
	got := Plan(repoLikeModels(), PlanOptions{
		Seq: 8192, Precision: "fp16", PerStep: 1,
		Steps: []string{"tensorrt", "onnx-cuda", "onnx-directml", "onnx-cpu"},
	})
	want := []struct{ step, name, provider string }{
		{"tensorrt", "laya_s8192_fp16_p2.engine", ""},
		{"onnx-cuda", "laya_ctx8192.opt.fp16.onnx", "cuda"},
		{"onnx-directml", "laya_ctx8192.onnx", "directml"},
		{"onnx-cpu", "laya_ctx8192.opt.onnx", "cpu"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d attempts: %+v", len(got), got)
	}
	for i, w := range want {
		if got[i].Step != w.step || !strings.HasSuffix(got[i].Path, w.name) || got[i].Provider != w.provider {
			t.Errorf("attempt %d: got %s %s %s, want %s %s %s", i,
				got[i].Step, filepath.Base(got[i].Path), got[i].Provider, w.step, w.name, w.provider)
		}
	}
}

func TestPlanNeverPicksShortOrUnlabelledPlan(t *testing.T) {
	for _, m := range Rank(repoLikeModels(), "tensorrt", PlanOptions{Seq: 8192, Precision: "fp16"}) {
		if m.SeqMax < 8192 {
			t.Errorf("picked %s with seq %d for an 8192 target", m.Name, m.SeqMax)
		}
	}
}

func TestPlanPinnedFirst(t *testing.T) {
	got := Plan(repoLikeModels(), PlanOptions{
		Seq: 8192, Precision: "fp16", Pinned: `D:\x\custom.engine`,
		Steps: []string{"tensorrt", "onnx-cpu"},
	})
	if len(got) == 0 || got[0].Path != `D:\x\custom.engine` || got[0].Step != "tensorrt" {
		t.Fatalf("pinned model not first: %+v", got)
	}
}

func TestConversionSourceSkipsFusedGraphs(t *testing.T) {
	m, ok := ConversionSource(repoLikeModels(), 8192)
	if !ok || m.Name != "laya_ctx8192.onnx" {
		t.Fatalf("got %v %q, want the plain graph", ok, m.Name)
	}
}

func TestBuildSpecTwoProfileArgs(t *testing.T) {
	s := BuildSpec{Source: "a.onnx", Seq: 8192, Precision: "fp16"}
	if n := s.OutputName(); n != "laya_s8192_fp16_p2.engine" {
		t.Fatalf("OutputName = %s", n)
	}
	args := strings.Join(s.Args("out.engine"), " ")
	for _, want := range []string{"--profile=0", "--profile=1", "input_ids:8x512", "input_ids:1x8192", "--fp16"} {
		if !strings.Contains(args, want) {
			t.Errorf("args missing %q: %s", want, args)
		}
	}
	short := BuildSpec{Source: "a.onnx", Seq: 512, Precision: "fp32"}
	if n := short.OutputName(); n != "laya_s512_fp32_b8.engine" {
		t.Fatalf("short OutputName = %s", n)
	}
	if a := strings.Join(short.Args("o"), " "); strings.Contains(a, "--profile=1") || strings.Contains(a, "--fp16") {
		t.Errorf("single-profile fp32 args wrong: %s", a)
	}
}

func TestCatalogUsesSidecarAndSeesNewFiles(t *testing.T) {
	dir := t.TempDir()
	eng := touch(t, dir, "custom.engine")
	if err := WriteSidecar(eng, Sidecar{SeqMax: 8192, BatchMax: 8, MultiProfile: true, Precision: "fp16"}); err != nil {
		t.Fatal(err)
	}
	c := NewCatalog([]string{dir})
	ms := c.Models()
	if len(ms) != 1 || ms[0].SeqMax != 8192 || !ms[0].MultiProfile || ms[0].Origin != "sidecar" {
		t.Fatalf("sidecar not applied: %+v", ms)
	}
	touch(t, dir, "laya_ctx8192.onnx")
	c.Invalidate()
	if n := len(c.Models()); n != 2 {
		t.Fatalf("after adding a file: %d models", n)
	}
}
