package modelstore

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// BuildSpec describes one TensorRT plan to build from an ONNX graph.
type BuildSpec struct {
	Source    string `json:"source"`
	OutDir    string `json:"out_dir"`
	Seq       int    `json:"seq"`
	Precision string `json:"precision"`
	// MarkersMax is the option ceiling per question.
	MarkersMax int `json:"markers_max"`
	// ShortSeq/ShortBatch shape the batched profile of a two-profile plan.
	ShortSeq   int `json:"short_seq"`
	ShortBatch int `json:"short_batch"`
	// BuilderOpt is TensorRT's builderOptimizationLevel.
	BuilderOpt   int    `json:"builder_opt"`
	WorkspaceMiB int    `json:"workspace_mib"`
	Trtexec      string `json:"trtexec"`
}

func (s *BuildSpec) defaults() {
	if s.Seq <= 0 {
		s.Seq = 8192
	}
	if s.Precision == "" {
		s.Precision = "fp16"
	}
	if s.MarkersMax <= 0 {
		s.MarkersMax = 64
	}
	if s.ShortSeq <= 0 {
		s.ShortSeq = 512
	}
	if s.ShortBatch <= 0 {
		s.ShortBatch = 8
	}
	if s.BuilderOpt <= 0 {
		s.BuilderOpt = 3
	}
	if s.WorkspaceMiB <= 0 {
		s.WorkspaceMiB = 6000
	}
}

// OutputName is the plan file this spec produces. It follows the naming
// bench/build-engines.ps1 uses, so the catalog reads it the same way.
func (s BuildSpec) OutputName() string {
	s.defaults()
	if s.Seq > s.ShortSeq {
		return fmt.Sprintf("laya_s%d_%s_p2.engine", s.Seq, s.Precision)
	}
	return fmt.Sprintf("laya_s%d_%s_b%d.engine", s.Seq, s.Precision, s.ShortBatch)
}

func profile(batch, seq, markers int) string {
	return fmt.Sprintf("input_ids:%dx%d,attention_mask:%dx%d,marker_pos:%dx%d,marker_mask:%dx%d,qtype:%d",
		batch, seq, batch, seq, batch, markers, batch, markers, batch)
}

// Args renders the trtexec command line (without the executable).
//
// For a ceiling above ShortSeq the plan gets two profiles: a batched one up to
// ShortSeq for the common short request, and a batch-1 one up to Seq. A single
// batched profile to 8192 does not build on a 12 GB card, and a batch-1-only
// plan pays a forward pass per question (bench/REPORT.md).
func (s BuildSpec) Args(outPath string) []string {
	s.defaults()
	min := profile(1, 64, 2)
	optM := 8
	if s.MarkersMax < 8 {
		optM = s.MarkersMax
	}
	args := []string{"--onnx=" + s.Source, "--saveEngine=" + outPath}
	if s.Seq > s.ShortSeq {
		longOpt := s.Seq
		if longOpt > 1024 {
			longOpt = 1024
		}
		args = append(args,
			"--profile=0",
			"--minShapes="+min,
			"--optShapes="+profile(1, s.ShortSeq/2, optM),
			"--maxShapes="+profile(s.ShortBatch, s.ShortSeq, s.MarkersMax),
			"--profile=1",
			"--minShapes="+min,
			"--optShapes="+profile(1, longOpt, optM),
			"--maxShapes="+profile(1, s.Seq, s.MarkersMax),
		)
	} else {
		opt := s.Seq / 2
		if opt < 64 {
			opt = 64
		}
		args = append(args,
			"--minShapes="+min,
			"--optShapes="+profile(1, opt, optM),
			"--maxShapes="+profile(s.ShortBatch, s.Seq, s.MarkersMax),
		)
	}
	args = append(args,
		fmt.Sprintf("--builderOptimizationLevel=%d", s.BuilderOpt),
		fmt.Sprintf("--memPoolSize=workspace:%d", s.WorkspaceMiB),
		"--skipInference",
	)
	if s.Precision == "fp16" {
		args = append(args, "--fp16")
	}
	return args
}

// FindTrtexec locates trtexec.exe: explicit path, TENSORRT_ROOT, C:\TensorRT-*,
// then PATH.
func FindTrtexec(explicit string) (string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", fmt.Errorf("trtexec: %w", err)
		}
		return explicit, nil
	}
	exe := "trtexec"
	if isWindows() {
		exe = "trtexec.exe"
	}
	var candidates []string
	if root := os.Getenv("TENSORRT_ROOT"); root != "" {
		candidates = append(candidates, filepath.Join(root, "bin", exe))
	}
	if matches, _ := filepath.Glob(`C:\TensorRT-*`); len(matches) > 0 {
		for i := len(matches) - 1; i >= 0; i-- {
			candidates = append(candidates, filepath.Join(matches[i], "bin", exe))
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	if p, err := exec.LookPath(exe); err == nil {
		return p, nil
	}
	return "", errors.New("trtexec not found (install TensorRT, set TENSORRT_ROOT or trtexec_path)")
}

func isWindows() bool { return os.PathSeparator == '\\' }

// Job is the state of one conversion.
type Job struct {
	ID        string    `json:"id"`
	Spec      BuildSpec `json:"spec"`
	Output    string    `json:"output"`
	Log       string    `json:"log"`
	State     string    `json:"state"` // queued, running, done, failed, canceled
	Error     string    `json:"error,omitempty"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`
	Seconds   float64   `json:"seconds"`
	// Tail holds the last lines of trtexec output, for a status page.
	Tail []string `json:"tail"`
}

// Converter runs at most one trtexec build at a time.
type Converter struct {
	mu     sync.Mutex
	jobs   []*Job
	cancel context.CancelFunc
	seq    int
	// OnDone runs after a job finishes (successfully or not).
	OnDone func(Job)
}

// NewConverter builds an idle converter.
func NewConverter() *Converter { return &Converter{} }

// Jobs lists every job, newest first.
func (c *Converter) Jobs() []Job {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Job, 0, len(c.jobs))
	for i := len(c.jobs) - 1; i >= 0; i-- {
		j := *c.jobs[i]
		j.Tail = append([]string(nil), j.Tail...)
		if j.State == "running" {
			j.Seconds = time.Since(j.StartedAt).Seconds()
		}
		out = append(out, j)
	}
	return out
}

// Running is the job in flight, if any.
func (c *Converter) Running() (Job, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, j := range c.jobs {
		if j.State == "running" || j.State == "queued" {
			return *j, true
		}
	}
	return Job{}, false
}

// Cancel stops the running build.
func (c *Converter) Cancel() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel == nil {
		return false
	}
	c.cancel()
	return true
}

// Start launches a build in the background. It refuses to start a second one,
// because two concurrent TensorRT builds exhaust a single GPU.
func (c *Converter) Start(spec BuildSpec) (Job, error) {
	spec.defaults()
	if _, err := os.Stat(spec.Source); err != nil {
		return Job{}, fmt.Errorf("convert: source: %w", err)
	}
	trtexec, err := FindTrtexec(spec.Trtexec)
	if err != nil {
		return Job{}, err
	}
	spec.Trtexec = trtexec
	if spec.OutDir == "" {
		spec.OutDir = filepath.Dir(spec.Source)
	}
	if err := os.MkdirAll(spec.OutDir, 0o755); err != nil {
		return Job{}, err
	}

	c.mu.Lock()
	for _, j := range c.jobs {
		if j.State == "running" || j.State == "queued" {
			c.mu.Unlock()
			return *j, fmt.Errorf("convert: a build is already running (%s)", j.Output)
		}
	}
	c.seq++
	out := filepath.Join(spec.OutDir, spec.OutputName())
	job := &Job{
		ID:     fmt.Sprintf("build-%d", c.seq),
		Spec:   spec,
		Output: out,
		// Logged beside the .partial file; renamed to <out>.build.log only on
		// success, so a cancelled or failed build never clobbers the log of
		// the plan already on disk.
		Log:       out + ".partial.log",
		State:     "running",
		StartedAt: time.Now(),
	}
	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	c.jobs = append(c.jobs, job)
	c.mu.Unlock()

	go c.run(ctx, job)
	return *job, nil
}

func (c *Converter) run(ctx context.Context, job *Job) {
	spec := job.Spec
	// Build into a temporary name, so a half-written plan is never picked up
	// by a scan and a failed build leaves the previous plan intact.
	tmp := job.Output + ".partial"
	_ = os.Remove(tmp)
	err := c.exec(ctx, job, spec.Trtexec, spec.Args(tmp))
	if err == nil {
		if st, serr := os.Stat(tmp); serr != nil || st.Size() == 0 {
			err = errors.New("trtexec reported success but wrote no engine")
		}
	}
	if err == nil {
		_ = os.Remove(job.Output)
		err = os.Rename(tmp, job.Output)
	}
	if err == nil {
		src, _ := os.Stat(spec.Source)
		sc := Sidecar{
			Source:       spec.Source,
			SeqMax:       spec.Seq,
			BatchMax:     spec.ShortBatch,
			MultiProfile: spec.Seq > spec.ShortSeq,
			Precision:    spec.Precision,
			BuiltAt:      time.Now().UTC(),
			BuildSeconds: time.Since(job.StartedAt).Seconds(),
		}
		if src != nil {
			sc.SourceSize, sc.SourceMTime = src.Size(), src.ModTime()
		}
		_ = WriteSidecar(job.Output, sc)
		final := job.Output + ".build.log"
		_ = os.Remove(final)
		if os.Rename(job.Log, final) == nil {
			c.mu.Lock()
			job.Log = final
			c.mu.Unlock()
		}
	} else {
		_ = os.Remove(tmp)
	}

	c.mu.Lock()
	job.EndedAt = time.Now()
	job.Seconds = job.EndedAt.Sub(job.StartedAt).Seconds()
	switch {
	case err == nil:
		job.State = "done"
	case ctx.Err() != nil:
		job.State = "canceled"
		job.Error = "canceled"
	default:
		job.State = "failed"
		job.Error = err.Error()
	}
	c.cancel = nil
	done := *job
	hook := c.OnDone
	c.mu.Unlock()
	if hook != nil {
		hook(done)
	}
}

func (c *Converter) exec(ctx context.Context, job *Job, bin string, args []string) error {
	logFile, err := os.Create(job.Log)
	if err != nil {
		return err
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "# %s %s\n", bin, strings.Join(args, " "))

	cmd := exec.CommandContext(ctx, bin, args...)
	pr, pw, err := os.Pipe()
	if err != nil {
		return err
	}
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		pw.Close()
		pr.Close()
		return err
	}
	pw.Close()
	sc := bufio.NewScanner(pr)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		fmt.Fprintln(logFile, line)
		c.mu.Lock()
		job.Tail = append(job.Tail, line)
		if len(job.Tail) > 40 {
			job.Tail = job.Tail[len(job.Tail)-40:]
		}
		c.mu.Unlock()
	}
	pr.Close()
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("trtexec: %w (log: %s)", err, job.Log)
	}
	return nil
}
