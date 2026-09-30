// Command engprobe reports what a TensorRT plan declares, without running
// inference.
//
// It answers the questions that decide how a plan should be loaded: the IO
// contract, the optimisation-profile bounds, and the worst-case activation
// memory per execution context. That last number is what makes a context count
// safe or reckless — a plan that needs several GB per context cannot be given
// eight of them on a laptop GPU.
//
// It creates no execution context and runs no inference, so it is safe to run
// while another process is using the GPU.
//
//	go run ./bench/engprobe -engine C:\path\to\model.engine
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/local/laya-go-launcher/internal/kernel"
)

func main() {
	enginePath := flag.String("engine", "", "engine file to inspect")
	freeHeadroom := flag.Int("headroom-mb", 512, "VRAM to leave free when suggesting a context count")
	flag.Parse()

	if *enginePath == "" {
		fmt.Fprintln(os.Stderr, "usage: engprobe -engine <path.engine>")
		os.Exit(2)
	}

	if err := kernel.Initialize(); err != nil {
		fmt.Fprintf(os.Stderr, "tensorrt: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("TensorRT   : %s\n", kernel.Version())

	dev := kernel.QueryDeviceInfo()
	vram := kernel.QueryVRAMInfo()
	fmt.Printf("device     : %s (sm_%d.%d)\n", dev.Name, dev.ComputeMajor, dev.ComputeMinor)
	fmt.Printf("VRAM       : %d MiB free / %d MiB total\n\n", vram.FreeMB, vram.TotalMB)

	eng, err := kernel.LoadEngine(*enginePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer eng.Close()

	tensors, err := eng.Tensors()
	if err != nil {
		fmt.Fprintf(os.Stderr, "tensors: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("engine     : %s\n\n", *enginePath)
	fmt.Printf("%-4s %-16s %-9s %s\n", "dir", "name", "dtype", "shape")
	for _, t := range tensors {
		dir := "out"
		if t.IsInput {
			dir = "in"
		}
		fmt.Printf("%-4s %-16s %-9s %v\n", dir, t.Name, t.DType, t.Shape)
	}

	activation := eng.DeviceMemoryBytes()
	activationMB := float64(activation) / (1024 * 1024)
	fmt.Printf("\nactivation memory per context : %.1f MiB (worst case)\n", activationMB)

	fmt.Printf("\noptimisation profiles:\n")
	for _, t := range tensors {
		if !t.IsInput {
			continue
		}
		for dim := range t.Shape {
			min, max, ok := eng.ProfileBounds(t.Name, dim)
			if !ok {
				continue
			}
			fmt.Printf("  %-16s dim %d : %d .. %d\n", t.Name, dim, min, max)
		}
	}

	// How many contexts actually fit, using the same accounting the manager
	// uses: each context needs its activation arena plus a staging allowance.
	fmt.Printf("\ncontext budget:\n")
	const stagingMiB = 64
	perContext := activationMB + stagingMiB
	available := float64(vram.FreeMB - *freeHeadroom)
	if available <= 0 {
		fmt.Printf("  only %d MiB free, below the %d MiB headroom: at most 1 context\n",
			vram.FreeMB, *freeHeadroom)
		return
	}
	n := int(available / perContext)
	if n < 1 {
		n = 1
	}
	fmt.Printf("  %.1f MiB per context (%.1f activation + %d staging)\n", perContext, activationMB, stagingMiB)
	fmt.Printf("  %.0f MiB usable after %d MiB headroom -> %d context(s) fit\n", available, *freeHeadroom, n)
	if n > 8 {
		fmt.Printf("  the manager caps this at 8\n")
	}
	if activationMB > available {
		fmt.Printf("  WARNING: one context alone exceeds the free VRAM; loading will likely fail\n")
	}
}
