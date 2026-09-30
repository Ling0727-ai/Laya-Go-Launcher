package engine

import (
	"testing"

	"github.com/local/laya-go-launcher/internal/kernel"
)

// Reference machine: RTX 5070 Ti Laptop, 12199 MiB total. The two measured
// plans are the ones that motivated this policy.
const (
	refTotalMB = 12199
	// laya_ctx8192.engine: profile allows 8192 tokens.
	bigActivationBytes = uint64(4563) * 1024 * 1024
	// laya_1k.engine: profile allows 1024 tokens.
	smallActivationBytes = uint64(88) * 1024 * 1024
)

// TestChooseContextsLargePlanDoesNotClaimTheGPU is the regression guard for the
// behaviour that was measured on the reference machine: a plan built for 8192
// tokens reserved 4563 MiB per context, and the old policy — free VRAM divided
// by per-context cost — handed out 2 contexts and ~9.1 GiB of a 12 GiB GPU for
// capacity no request used (max_len was 512).
func TestChooseContextsLargePlanDoesNotClaimTheGPU(t *testing.T) {
	vram := kernel.VRAMInfo{FreeMB: 11042, TotalMB: refTotalMB}

	n := chooseContexts(bigActivationBytes, vram, headroomMiB, maxVRAMShare, maxContexts)

	// Total VRAM share is 0.5 * 12199 = 6099 MiB, minus one activation arena
	// (4563) = 1536 MiB, so one context fits and two do not.
	if n != 1 {
		t.Errorf("large plan: chose %d contexts, want 1 (the share budget allows one)", n)
	}

	usedMiB := float64(n) * (float64(bigActivationBytes)/(1024*1024) + stagingMiB)
	if usedMiB > float64(refTotalMB)*maxVRAMShare {
		t.Errorf("large plan: %d contexts would use %.0f MiB, over the %.0f MiB share budget",
			n, usedMiB, float64(refTotalMB)*maxVRAMShare)
	}
}

// TestChooseContextsSmallPlanAllowsConcurrency is the other half: the same
// policy must not throttle a plan that is genuinely cheap, or the fix would
// trade one problem for another.
func TestChooseContextsSmallPlanAllowsConcurrency(t *testing.T) {
	vram := kernel.VRAMInfo{FreeMB: 11042, TotalMB: refTotalMB}

	n := chooseContexts(smallActivationBytes, vram, headroomMiB, maxVRAMShare, maxContexts)

	if n != maxContexts {
		t.Errorf("small plan: chose %d contexts, want the cap %d", n, maxContexts)
	}
}

func TestChooseContextsCases(t *testing.T) {
	cases := []struct {
		name        string
		activation  uint64
		vram        kernel.VRAMInfo
		headroom    float64
		share       float64
		cap         int
		want        int
	}{
		{
			name: "no VRAM reported at all",
			activation: smallActivationBytes, vram: kernel.VRAMInfo{},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 1,
		},
		{
			name: "free VRAM below the headroom reserve",
			activation: smallActivationBytes, vram: kernel.VRAMInfo{FreeMB: 256, TotalMB: refTotalMB},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 1,
		},
		{
			name: "activation alone exceeds free VRAM",
			activation: bigActivationBytes, vram: kernel.VRAMInfo{FreeMB: 2048, TotalMB: refTotalMB},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 1,
		},
		{
			name: "free VRAM is the binding constraint",
			// 1024 MiB free - 512 headroom = 512 usable; 88+64 = 152 per context
			// -> 3 fit, below the share budget and the cap.
			activation: smallActivationBytes, vram: kernel.VRAMInfo{FreeMB: 1024, TotalMB: refTotalMB},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 3,
		},
		{
			name: "the cap binds when everything is roomy",
			activation: smallActivationBytes, vram: kernel.VRAMInfo{FreeMB: 11042, TotalMB: refTotalMB},
			headroom: headroomMiB, share: maxVRAMShare, cap: 2,
			want: 2,
		},
		{
			name: "share budget binds when total is small",
			// 0.5 * 1024 = 512, minus 152 activation = 360 usable -> 2 fit
			// (88+64 = 152 each), while free VRAM alone would allow more.
			activation: smallActivationBytes, vram: kernel.VRAMInfo{FreeMB: 8192, TotalMB: 1024},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 2,
		},
		{
			name: "a plan needing zero activation still yields at least one",
			activation: 0, vram: kernel.VRAMInfo{FreeMB: 11042, TotalMB: refTotalMB},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: maxContexts,
		},
		{
			name: "zero activation with no free VRAM still yields one",
			activation: 0, vram: kernel.VRAMInfo{FreeMB: 0, TotalMB: 0},
			headroom: headroomMiB, share: maxVRAMShare, cap: maxContexts,
			want: 1,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := chooseContexts(c.activation, c.vram, c.headroom, c.share, c.cap)
			if got != c.want {
				t.Errorf("chooseContexts = %d, want %d", got, c.want)
			}
			if got < 1 {
				t.Errorf("must always return at least 1, got %d", got)
			}
			if c.cap > 0 && got > c.cap {
				t.Errorf("returned %d, over the cap %d", got, c.cap)
			}
		})
	}
}

// TestChooseContextsNeverExceedsTotalShare states the invariant the policy
// exists to enforce, across a spread of plan sizes: whatever is chosen must fit
// inside the total-VRAM share, unless a single context cannot (in which case one
// is still returned so the caller fails on a real allocation error rather than a
// silent zero).
func TestChooseContextsNeverExceedsTotalShare(t *testing.T) {
	vram := kernel.VRAMInfo{FreeMB: refTotalMB, TotalMB: refTotalMB}
	shareMiB := float64(refTotalMB) * maxVRAMShare

	for activationMiB := 1; activationMiB <= 12000; activationMiB += 137 {
		activation := uint64(activationMiB) * 1024 * 1024
		n := chooseContexts(activation, vram, headroomMiB, maxVRAMShare, maxContexts)
		if n < 1 {
			t.Fatalf("activation=%d MiB: returned %d", activationMiB, n)
		}
		if n == 1 {
			continue // one context may exceed the share; the load will report it
		}
		used := float64(n) * (float64(activationMiB) + stagingMiB)
		if used > shareMiB {
			t.Errorf("activation=%d MiB: %d contexts use %.0f MiB, over the %.0f MiB share",
				activationMiB, n, used, shareMiB)
		}
	}
}
