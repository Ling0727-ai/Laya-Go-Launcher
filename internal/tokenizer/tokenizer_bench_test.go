package tokenizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/text/unicode/norm"
)

// nfcString is the normalisation the pipeline applies first, named so tests can
// build a reference encoding without duplicating the import.
func nfcString(s string) string { return norm.NFC.String(s) }

// These benchmarks decompose the encode pipeline so the expensive stage is
// identified by measurement rather than assumed. The stages are the ones
// encodeNoSpecials runs in order: NFC, the added-token split, the GPT-2 split,
// the byte-level map, and BPE.

func benchText(repeats int) string {
	return strings.Repeat("We were billed twice for March. Please refund the duplicate today. ", repeats)
}

func benchTokenizer(b *testing.B) *Tokenizer {
	b.Helper()
	if p := lookupTokenizerPathForBench(); p != "" {
		tk, err := Load(p)
		if err != nil {
			b.Fatalf("load tokenizer: %v", err)
		}
		return tk
	}
	b.Skip("laya tokenizer.json not found")
	return nil
}

func BenchmarkStageNFC(b *testing.B) {
	text := benchText(120)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = norm.NFC.String(text)
	}
}

func BenchmarkStageSplitOnAdded(b *testing.B) {
	tk := benchTokenizer(b)
	text := norm.NFC.String(benchText(120))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.splitOnAdded(text)
	}
}

func BenchmarkStageGPT2Split(b *testing.B) {
	rs := []rune(benchText(120))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gpt2Split(rs)
	}
}

func BenchmarkStageByteLevelMap(b *testing.B) {
	piece := "Ġbilled Ġtwice Ġfor ĠMarch."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = byteLevelMap(piece)
	}
}

// BenchmarkFullEncode is the whole pipeline on the same text, so each stage's
// share is readable against it.
func BenchmarkFullEncode(b *testing.B) {
	tk := benchTokenizer(b)
	text := benchText(120)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode(text)
	}
}

// BenchmarkFullEncodeBounded is the same text with a token budget, showing how
// much of the pipeline the bound can actually skip.
func BenchmarkFullEncodeBounded(b *testing.B) {
	tk := benchTokenizer(b)
	text := benchText(120)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.EncodePrefix(text, 512)
	}
}

// BenchmarkFullEncodeDistinct uses text with no repeated pre-tokens, so the BPE
// memoisation cannot hide merge cost. A real document looks more like this than
// like a repeated sentence.
func BenchmarkFullEncodeDistinct(b *testing.B) {
	tk := benchTokenizer(b)
	var sb strings.Builder
	for i := 0; sb.Len() < 8000; i++ {
		sb.WriteString("token")
		sb.WriteString(strings.Repeat("x", i%7+1))
		sb.WriteString(" value")
		sb.WriteString(strings.Repeat("y", (i*3)%5+1))
		sb.WriteString(" ")
	}
	text := sb.String()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode(text)
	}
}

func lookupTokenizerPathForBench() string {
	// Same discovery as the tests, duplicated because the test helper takes a
	// *testing.T rather than the *testing.B these benchmarks have.
	if p := os.Getenv("LAYA_TOKENIZER"); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", "tokenizer", "tokenizer.json")
	matches, err := filepath.Glob(pattern)
	if err != nil || len(matches) == 0 {
		return ""
	}
	return matches[0]
}
