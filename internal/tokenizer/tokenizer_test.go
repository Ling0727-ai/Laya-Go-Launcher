package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// tokenizerPath locates the laya tokenizer.json in the HF cache.
func tokenizerPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_TOKENIZER"); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory")
	}
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", "tokenizer", "tokenizer.json")
	matches, err := filepath.Glob(pattern)
	if err != nil || len(matches) == 0 {
		t.Skipf("laya tokenizer.json not found under %s", pattern)
	}
	return matches[0]
}

func loadTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tk, err := Load(tokenizerPath(t))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return tk
}

func TestSpecialIDs(t *testing.T) {
	tk := loadTokenizer(t)
	if tk.CLSID() != 50281 {
		t.Errorf("CLS id = %d, want 50281", tk.CLSID())
	}
	if tk.SEPID() != 50282 {
		t.Errorf("SEP id = %d, want 50282", tk.SEPID())
	}
	if tk.PadID() != 50283 {
		t.Errorf("PAD id = %d, want 50283", tk.PadID())
	}
	if tk.MaskID() != 50284 {
		t.Errorf("MASK id = %d, want 50284", tk.MaskID())
	}
	if tk.VocabSize() < 50300 {
		t.Errorf("vocab size = %d, want >= 50300", tk.VocabSize())
	}
}

func TestBasicEncoding(t *testing.T) {
	tk := loadTokenizer(t)

	// Round-trip: decoding an encoding must reproduce the input for text that
	// survives NFC unchanged.
	for _, text := range []string{
		"hello world",
		"Duplicate charge on invoice #4411",
		"We were billed twice for March.",
		"user@acme.com",
	} {
		ids := tk.Encode(text)
		if len(ids) == 0 {
			t.Errorf("Encode(%q) produced no tokens", text)
			continue
		}
		if got := tk.Decode(ids); got != text {
			t.Errorf("round trip %q -> %q", text, got)
		}
	}
}

func TestWithSpecials(t *testing.T) {
	tk := loadTokenizer(t)
	plain := tk.Encode("hello")
	wrapped := tk.EncodeWithSpecials("hello")

	if len(wrapped) != len(plain)+2 {
		t.Fatalf("wrapped length %d, want %d", len(wrapped), len(plain)+2)
	}
	if wrapped[0] != tk.CLSID() {
		t.Errorf("first id = %d, want CLS %d", wrapped[0], tk.CLSID())
	}
	if wrapped[len(wrapped)-1] != tk.SEPID() {
		t.Errorf("last id = %d, want SEP %d", wrapped[len(wrapped)-1], tk.SEPID())
	}
}

func TestMaskTokenIsSingle(t *testing.T) {
	tk := loadTokenizer(t)
	ids := tk.Encode("[MASK]")
	if len(ids) != 1 || ids[0] != tk.MaskID() {
		t.Errorf("Encode(\"[MASK]\") = %v, want [%d]", ids, tk.MaskID())
	}
}

// TestParityAgainstPython is the test that matters: it runs the real Python
// tokenizer over the same strings and compares ids exactly. Any drift in the
// byte-level table, the split regex, or the merge order shows up here.
func TestParityAgainstPython(t *testing.T) {
	python := os.Getenv("LAYA_PYTHON")
	if python == "" {
		home, _ := os.UserHomeDir()
		candidate := filepath.Join(home, "Documents", "laya", ".venv", "Scripts", "python.exe")
		if _, err := os.Stat(candidate); err != nil {
			t.Skip("set LAYA_PYTHON to a python with `tokenizers` installed")
		}
		python = candidate
	}
	tokPath := tokenizerPath(t)

	inputs := []string{
		"hello world",
		"Duplicate charge on invoice #4411",
		"We were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
		"user@acme.com",
		"Which department should handle this request?",
		"billing: invoices, payments, refunds",
		"Urgent!!! 3 items missing (order #A-19).",
		"  leading and trailing  ",
		"tab\tseparated\tvalues",
		"newline\nseparated",
		"camelCaseIdentifier and snake_case_name",
		"Numbers 1234567890 and 3.14159",
		"Ünïcödé with diacritics: café, naïve, Zürich",
		"日本語のテキスト",
		"Русский текст",
		"العربية نص",
		"Emoji 🚀🔥 and symbols ©®™",
		"[MASK] [CLS] [SEP] [PAD]",
		"",
		"a",
	}

	// The reference script prints one JSON array of ids per input line.
	script := `
import json, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
out = []
for line in sys.stdin.read().split("\n\x1e"):
    out.append(tk.encode(line, add_special_tokens=False).ids)
print(json.dumps(out))
`
	tmp := filepath.Join(t.TempDir(), "ref.py")
	if err := os.WriteFile(tmp, []byte(script), 0o644); err != nil {
		t.Fatalf("write reference script: %v", err)
	}

	var stdin strings.Builder
	for i, s := range inputs {
		if i > 0 {
			stdin.WriteString("\n\x1e")
		}
		stdin.WriteString(s)
	}

	cmd := exec.Command(python, tmp, tokPath)
	cmd.Stdin = strings.NewReader(stdin.String())
	raw, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			t.Skipf("python reference unavailable: %s", string(ee.Stderr))
		}
		t.Fatalf("run python reference: %v", err)
	}

	var want [][]int
	if err := json.Unmarshal([]byte(lastLine(string(raw))), &want); err != nil {
		t.Fatalf("parse python output: %v\n%s", err, raw)
	}
	if len(want) != len(inputs) {
		t.Fatalf("python returned %d results for %d inputs", len(want), len(inputs))
	}

	tk := loadTokenizer(t)
	for i, text := range inputs {
		got := tk.Encode(text)
		if !equalInts(got, want[i]) {
			t.Errorf("mismatch for %q\n  go:     %v\n  python: %v", text, got, want[i])
		}
	}
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return s[i+1:]
	}
	return s
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestMergeParsing checks both tokenizer.json merge shapes parse.
func TestMergeParsing(t *testing.T) {
	tk := loadTokenizer(t)
	if len(tk.ranks) < 40000 {
		t.Errorf("parsed %d merges, want >= 40000", len(tk.ranks))
	}
}

// TestCacheConsistency checks the memoised path returns the same ids.
func TestCacheConsistency(t *testing.T) {
	tk := loadTokenizer(t)
	const text = "Duplicate charge on invoice #4411"
	first := tk.Encode(text)
	for i := 0; i < 5; i++ {
		if got := tk.Encode(text); !equalInts(got, first) {
			t.Fatalf("cache changed the result: %v vs %v", got, first)
		}
	}
}

// TestConcurrentEncode exercises the cache under -race.
func TestConcurrentEncode(t *testing.T) {
	tk := loadTokenizer(t)
	texts := []string{"hello world", "billing: invoices", "We were billed twice"}
	done := make(chan bool)
	for w := 0; w < 8; w++ {
		go func(w int) {
			defer func() { done <- true }()
			for i := 0; i < 50; i++ {
				tk.Encode(texts[(w+i)%len(texts)])
			}
		}(w)
	}
	for w := 0; w < 8; w++ {
		<-done
	}
}

var _ = fmt.Sprintf
var _ = bufio.NewReader
