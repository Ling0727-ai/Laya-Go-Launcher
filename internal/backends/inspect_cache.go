package backends

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/local/laya-go-launcher/internal/backend"
)

// inspectionCache memoises Inspect results per model file.
//
// Reading a model's IO contract is the only reliable way to tell a laya model
// from any other file, but it is not cheap: a plan has to be deserialized (its
// weights land on the GPU) and a graph needs a session built. The engine picker
// scans a directory on every refresh, and the GUI refreshes after every load and
// unload, so without a cache the same five plans would be deserialized five
// times per click — while a model is already resident and holding the GPU.
//
// A hit is keyed by path, size and modification time, so editing or replacing a
// file invalidates its entry without any explicit call. Entries also expire, so
// a long-lived process does not accumulate results for files that no longer
// exist, and a negative result is cached too: a broken file is exactly the one a
// rescan should not re-open.
type inspectionCache struct {
	mu      sync.Mutex
	entries map[string]inspectionEntry
	ttl     time.Duration
	now     func() time.Time
}

type inspectionEntry struct {
	info    backend.Info
	err     error
	modTime time.Time
	size    int64
	at      time.Time
}

// inspectionTTL is how long a cached inspection stays valid. It only has to
// outlive a picker refresh; a minute is long enough that a scan is free and
// short enough that a stale entry cannot mislead for long.
const inspectionTTL = time.Minute

// inspectCache is the process-wide cache. One is enough: inspections are pure
// reads of a file's metadata, so two callers asking about the same path want the
// same answer regardless of which kernel configuration they carry.
var inspectCache = &inspectionCache{
	entries: make(map[string]inspectionEntry),
	ttl:     inspectionTTL,
	now:     time.Now,
}

// get returns a cached inspection of path, computing it with load on a miss.
//
// The lock is held across load on purpose. Inspections are serialized anyway —
// they touch the GPU and the native runtimes — and holding it means N concurrent
// callers asking about one path produce one load rather than N. The failure mode
// of not holding it is precisely the stampede this cache exists to prevent.
func (c *inspectionCache) get(ctx context.Context, cfg Config, path string, load func() (backend.Info, error)) (backend.Info, error) {
	key := cacheKey(path)
	stat, statErr := os.Stat(path)
	if statErr != nil {
		// A file that cannot be stat'd cannot be inspected either; report the
		// stat error directly rather than letting a kernel produce a vaguer one.
		return backend.Info{}, statErr
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if e, ok := c.entries[key]; ok && c.valid(e, stat) {
		return e.info, e.err
	}

	info, err := load()
	c.entries[key] = inspectionEntry{
		info:    info,
		err:     err,
		modTime: stat.ModTime(),
		size:    stat.Size(),
		at:      c.now(),
	}
	// Keep the map from growing without bound in a process that has seen many
	// models: a stale entry is worthless, so pruning is free.
	c.pruneLocked()
	return info, err
}

// valid reports whether a cached entry still describes the file on disk.
func (c *inspectionCache) valid(e inspectionEntry, stat os.FileInfo) bool {
	if c.now().Sub(e.at) > c.ttl {
		return false
	}
	return e.modTime.Equal(stat.ModTime()) && e.size == stat.Size()
}

// pruneLocked drops expired entries. The caller holds the lock.
func (c *inspectionCache) pruneLocked() {
	for k, e := range c.entries {
		if c.now().Sub(e.at) > c.ttl {
			delete(c.entries, k)
		}
	}
}

// cacheKey normalises a path so two spellings of one file share an entry.
func cacheKey(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		return abs
	}
	return path
}

// InvalidateInspection forgets what was cached for a path, so the next Inspect
// re-reads the file.
//
// A replaced model usually changes size and mtime, which the key already
// detects. This exists for the case where it does not — a file restored from a
// backup with its timestamps preserved — and for tests.
func InvalidateInspection(path string) {
	inspectCache.mu.Lock()
	delete(inspectCache.entries, cacheKey(path))
	inspectCache.mu.Unlock()
}
