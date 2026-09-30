// Package engine owns the loaded plan and the memory that serves it.
//
// This is the part of the port that carries the QualityScaler-go allocation
// strategy: a context pool pre-allocated at load time and handed out as
// borrows, per-context pinned host buffers registered with the native side
// once, a size-keyed sync.Pool for scratch, and a bounded per-shape pair pool.
// Keeping allocation out of the per-run path is what makes latency stable.
package engine

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/local/laya-go-launcher/internal/kernel"
)

// ErrNotLoaded is returned by every operation that needs a plan.
var ErrNotLoaded = errors.New("engine: no engine is loaded")

// ErrClosed is returned once a manager has been shut down.
var ErrClosed = errors.New("engine: manager is closed")

// Info describes a loaded plan.
type Info struct {
	Path               string
	Contexts           int
	ActivationMemoryMB float64
	Inputs             []kernel.TensorInfo
	Outputs            []kernel.TensorInfo
}

// Options tune loading.
type Options struct {
	// Path is the engine file to load.
	Path string
	// Contexts is how many execution contexts to pre-allocate. Zero or negative
	// picks a count from free VRAM.
	Contexts int
}

// borrow is one checked-out execution context.
type borrow struct {
	ctx     *kernel.Context
	release func()
	once    sync.Once
}

// Close returns the context to the pool. Idempotent.
func (b *borrow) Close() {
	if b == nil {
		return
	}
	b.once.Do(func() {
		if b.release != nil {
			b.release()
		}
	})
}

// contextPool owns the native contexts. Contexts are destroyed only after every
// outstanding borrow is returned, so no caller can outlive its context.
type contextPool struct {
	mu        sync.Mutex
	available []*kernel.Context
	inUse     int
	closed    bool
	changed   chan struct{}
}

func newContextPool(ctxs []*kernel.Context) *contextPool {
	return &contextPool{available: ctxs, changed: make(chan struct{})}
}

func (p *contextPool) notifyLocked() {
	close(p.changed)
	p.changed = make(chan struct{})
}

func (p *contextPool) acquire(ctx context.Context) (*borrow, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if p.closed {
			return nil, ErrClosed
		}
		if n := len(p.available); n > 0 {
			c := p.available[n-1]
			p.available = p.available[:n-1]
			p.inUse++
			b := &borrow{ctx: c}
			b.release = func() {
				p.mu.Lock()
				p.available = append(p.available, c)
				p.inUse--
				p.notifyLocked()
				p.mu.Unlock()
			}
			return b, nil
		}
		changed := p.changed
		p.mu.Unlock()
		select {
		case <-ctx.Done():
		case <-changed:
		}
		p.mu.Lock()
	}
}

func (p *contextPool) closeAndDrain() []*kernel.Context {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	p.notifyLocked()
	for p.inUse > 0 {
		changed := p.changed
		p.mu.Unlock()
		<-changed
		p.mu.Lock()
	}
	out := p.available
	p.available = nil
	return out
}

// Resources holds the reusable host allocations for one engine.
//
// Scratch buffers are pooled by exact size and the pool map is bounded, so a
// pathological spread of sizes cannot grow without limit. Tensor pairs are
// pooled per shape with a creation limit, so concurrent workers share a bounded
// set instead of each allocating its own.
type Resources struct {
	scratchMu    sync.Mutex
	scratchPools map[int]*sync.Pool
	maxScratch   int

	pairMu    sync.Mutex
	pairPools map[pairKey]*pairPool
	maxShapes int
}

type pairKey struct {
	inputBytes  int
	outputBytes int
}

type pairPool struct {
	mu      sync.Mutex
	cond    *sync.Cond
	idle    [][2][]byte
	created int
	limit   int
}

// NewResources builds an empty allocator set.
func NewResources() *Resources {
	return &Resources{
		scratchPools: make(map[int]*sync.Pool),
		maxScratch:   128,
		pairPools:    make(map[pairKey]*pairPool),
		maxShapes:    16,
	}
}

// AcquireBytes returns a buffer of exactly size bytes from the size pool.
func (r *Resources) AcquireBytes(size int) []byte {
	if size <= 0 {
		return nil
	}
	r.scratchMu.Lock()
	pool := r.scratchPools[size]
	if pool == nil && len(r.scratchPools) < r.maxScratch {
		pool = &sync.Pool{New: func() any { return make([]byte, size) }}
		r.scratchPools[size] = pool
	}
	r.scratchMu.Unlock()
	if pool == nil {
		return make([]byte, size)
	}
	v, _ := pool.Get().([]byte)
	if cap(v) < size {
		return make([]byte, size)
	}
	return v[:size]
}

// ReleaseBytes returns a buffer to its capacity pool.
func (r *Resources) ReleaseBytes(v []byte) {
	if len(v) == 0 {
		return
	}
	r.scratchMu.Lock()
	pool := r.scratchPools[cap(v)]
	r.scratchMu.Unlock()
	if pool != nil {
		pool.Put(v[:cap(v)])
	}
}

// AcquirePair hands out a matched input/output buffer set for one shape,
// blocking while the pool is at its limit. The release function returns it.
func (r *Resources) AcquirePair(ctx context.Context, inputBytes, outputBytes, limit int) ([2][]byte, func(), error) {
	if inputBytes <= 0 || outputBytes <= 0 {
		return [2][]byte{}, nil, fmt.Errorf("engine: invalid pair sizes %d/%d", inputBytes, outputBytes)
	}
	key := pairKey{inputBytes, outputBytes}

	r.pairMu.Lock()
	pool := r.pairPools[key]
	if pool == nil && len(r.pairPools) < r.maxShapes {
		pool = &pairPool{limit: max(1, limit)}
		pool.cond = sync.NewCond(&pool.mu)
		r.pairPools[key] = pool
	}
	r.pairMu.Unlock()

	if pool == nil {
		// Too many distinct shapes in flight: allocate without pooling.
		return [2][]byte{r.AcquireBytes(inputBytes), r.AcquireBytes(outputBytes)},
			func() {}, nil
	}

	stop := context.AfterFunc(ctx, func() {
		pool.mu.Lock()
		pool.cond.Broadcast()
		pool.mu.Unlock()
	})
	defer stop()

	pool.mu.Lock()
	for len(pool.idle) == 0 && pool.created >= pool.limit {
		if err := ctx.Err(); err != nil {
			pool.mu.Unlock()
			return [2][]byte{}, nil, err
		}
		pool.cond.Wait()
	}
	if err := ctx.Err(); err != nil {
		pool.mu.Unlock()
		return [2][]byte{}, nil, err
	}
	if n := len(pool.idle); n > 0 {
		pair := pool.idle[n-1]
		pool.idle = pool.idle[:n-1]
		pool.mu.Unlock()
		return pair, func() {
			pool.mu.Lock()
			pool.idle = append(pool.idle, pair)
			pool.cond.Signal()
			pool.mu.Unlock()
		}, nil
	}
	pool.created++
	pool.mu.Unlock()

	pair := [2][]byte{make([]byte, inputBytes), make([]byte, outputBytes)}
	return pair, func() {
		pool.mu.Lock()
		pool.idle = append(pool.idle, pair)
		pool.cond.Signal()
		pool.mu.Unlock()
	}, nil
}

// ReleasePairs drains every shape pool, waiting for outstanding borrows.
func (r *Resources) ReleasePairs() {
	r.pairMu.Lock()
	pools := r.pairPools
	r.pairPools = make(map[pairKey]*pairPool)
	r.pairMu.Unlock()
	for _, p := range pools {
		p.mu.Lock()
		for len(p.idle) < p.created {
			p.cond.Wait()
		}
		p.idle = nil
		p.created = 0
		p.mu.Unlock()
	}
}

// ResetScratch drops every size pool.
func (r *Resources) ResetScratch() {
	r.scratchMu.Lock()
	r.scratchPools = make(map[int]*sync.Pool)
	r.scratchMu.Unlock()
}

// Manager owns the process's single loaded engine. It is the one place that
// serialises load and unload, so the rest of the program can treat the engine
// as either present or absent.
type Manager struct {
	mu       sync.RWMutex
	engine   *kernel.Engine
	pool     *contextPool
	info     Info
	res      *Resources
	loadedAt time.Time
}

// NewManager returns a manager with nothing loaded.
func NewManager() *Manager {
	return &Manager{res: NewResources()}
}

// Load reads a plan and pre-allocates its contexts, replacing any current
// engine. The previous engine is drained before the new one is created, so peak
// VRAM is not two engines.
func (m *Manager) Load(opts Options) (Info, error) {
	if opts.Path == "" {
		return Info{}, fmt.Errorf("engine: path is required")
	}
	if _, err := os.Stat(opts.Path); err != nil {
		return Info{}, fmt.Errorf("engine: %w", err)
	}
	if err := kernel.Initialize(); err != nil {
		return Info{}, err
	}

	// Unload outside the write lock's critical section is not possible without
	// losing the invariant, so this is done first and then re-taken.
	m.unload()

	eng, err := kernel.LoadEngine(opts.Path)
	if err != nil {
		return Info{}, err
	}
	tensors, err := eng.Tensors()
	if err != nil {
		eng.Close()
		return Info{}, err
	}

	n := opts.Contexts
	if n <= 0 {
		n = suggestContexts(eng)
	}
	ctxs := make([]*kernel.Context, 0, n)
	for i := 0; i < n; i++ {
		c, err := eng.NewContext()
		if err != nil {
			for _, created := range ctxs {
				created.Close()
			}
			eng.Close()
			return Info{}, fmt.Errorf("engine: pre-allocate context %d/%d: %w", i+1, n, err)
		}
		ctxs = append(ctxs, c)
	}

	info := Info{
		Path:               opts.Path,
		Contexts:           n,
		ActivationMemoryMB: float64(eng.DeviceMemoryBytes()) / (1024 * 1024),
	}
	for _, t := range tensors {
		if t.IsInput {
			info.Inputs = append(info.Inputs, t)
		} else {
			info.Outputs = append(info.Outputs, t)
		}
	}

	m.mu.Lock()
	m.engine = eng
	m.pool = newContextPool(ctxs)
	m.info = info
	m.loadedAt = time.Now()
	m.mu.Unlock()
	return info, nil
}

// suggestContexts bounds the pool by free VRAM. Each context needs its
// activation arena plus its pinned staging, and the engine reports the
// worst-case activation size.
//
// Two limits apply, and the smaller wins:
//
//   - free VRAM minus a headroom reserve, so loading does not starve the
//     desktop compositor or another process;
//   - a fraction of *total* VRAM, because "whatever is free right now" is a
//     poor budget on a laptop where other applications share the GPU.
//
// Measured on the reference machine, one context of a plan built for 8192
// tokens needs 4563 MiB while the same model built for 1024 needs 88 MiB. Filling
// free VRAM with contexts of the large plan therefore costs most of the GPU for
// capacity no request uses, which is why the total-VRAM share matters as much as
// the free-VRAM test.
func suggestContexts(eng *kernel.Engine) int {
	return chooseContexts(
		eng.DeviceMemoryBytes(),
		kernel.QueryVRAMInfo(),
		headroomMiB,
		maxVRAMShare,
		maxContexts,
	)
}

// Budget constants for chooseContexts.
const (
	// headroomMiB is left free for the desktop and other processes.
	headroomMiB = 512
	// maxVRAMShare caps how much of the *total* VRAM this process may claim for
	// activation arenas. One half leaves room for the engine weights, the
	// staging buffers, and everything else on the machine.
	maxVRAMShare = 0.5
	// maxContexts caps the pool regardless of how much VRAM is available.
	// Concurrency beyond this does not help an interactive workload, and each
	// context is real memory.
	maxContexts = 8
	// stagingMiB is the per-context allowance for pinned host buffers and IO
	// scratch, which are not part of the activation arena.
	stagingMiB = 64
)

// chooseContexts is suggestContexts as a pure function, so the policy can be
// tested without a GPU. activationBytes is the plan's worst-case activation
// memory for one context; vram is what the driver reports.
//
// headroomMiB is a reserve subtracted from free VRAM; share is the fraction of
// total VRAM this process may use; cap bounds the result.
func chooseContexts(activationBytes uint64, vram kernel.VRAMInfo, headroomMiB float64, share float64, cap int) int {
	activationMiB := float64(activationBytes) / (1024 * 1024)
	perContext := activationMiB + stagingMiB
	if perContext <= 0 {
		perContext = 1
	}

	// Budget 1: free VRAM minus the reserve. This is what is actually available
	// right now, but it fluctuates with whatever else is running.
	usable := float64(vram.FreeMB) - headroomMiB

	// Budget 2: a share of total VRAM. This is stable when other applications
	// hold memory, so it keeps one large plan from claiming the whole GPU.
	// Only meaningful when the driver reported a total.
	if vram.TotalMB > 0 {
		byShare := float64(vram.TotalMB)*share - activationMiB
		if byShare < usable {
			usable = byShare
		}
	}

	if usable <= 0 {
		return 1
	}
	n := int(usable / perContext)
	if n < 1 {
		return 1
	}
	if n > cap {
		return cap
	}
	return n
}

// unload drains and releases the current engine, if any.
func (m *Manager) unload() {
	m.mu.Lock()
	pool, eng := m.pool, m.engine
	m.pool, m.engine = nil, nil
	m.info = Info{}
	m.mu.Unlock()

	if pool != nil {
		for _, c := range pool.closeAndDrain() {
			c.Close()
		}
	}
	if eng != nil {
		eng.Close()
	}
	if m.res != nil {
		m.res.ReleasePairs()
	}
}

// Unload releases the engine. Idempotent.
func (m *Manager) Unload() { m.unload() }

// Close releases everything and marks the manager unusable.
func (m *Manager) Close() { m.unload() }

// Info describes the loaded engine, or a zero value when none is loaded.
func (m *Manager) Info() (Info, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.engine == nil {
		return Info{}, false
	}
	return m.info, true
}

// Loaded reports whether a plan is resident.
func (m *Manager) Loaded() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.engine != nil
}

// InputBounds returns the (min, max) range of one dimension of an input.
//
// For a fixed-shape plan min == max; for a dynamic one these come from the
// optimisation profile. Either way a caller gets the range the engine actually
// accepts, instead of having to know how the plan was built.
func (m *Manager) InputBounds(name string, dim int) (min, max int, ok bool) {
	m.mu.RLock()
	eng, info := m.engine, m.info
	m.mu.RUnlock()
	if eng == nil {
		return 0, 0, false
	}

	if lo, hi, dynamic := eng.ProfileBounds(name, dim); dynamic {
		return lo, hi, true
	}
	for _, t := range info.Inputs {
		if t.Name == name && dim < len(t.Shape) && t.Shape[dim] > 0 {
			return t.Shape[dim], t.Shape[dim], true
		}
	}
	return 0, 0, false
}

// Dir is the directory the loaded engine came from, which is where its
// checkpoint configuration (rl_agent_config.json) is expected to sit.
func (m *Manager) Dir() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.info.Path == "" {
		return ""
	}
	return filepath.Dir(m.info.Path)
}

// Resources exposes the host allocators.
func (m *Manager) Resources() *Resources { return m.res }

// WithContext borrows a context for the duration of fn.
func (m *Manager) WithContext(ctx context.Context, fn func(*kernel.Context) error) error {
	m.mu.RLock()
	pool := m.pool
	m.mu.RUnlock()
	if pool == nil {
		return ErrNotLoaded
	}
	b, err := pool.acquire(ctx)
	if err != nil {
		return err
	}
	defer b.Close()
	return fn(b.ctx)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
