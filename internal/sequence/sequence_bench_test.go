package sequence

import (
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/tokenizer"
)

// loadTokB is the *testing.B twin of loadTok, which is typed for *testing.T.
// Both skip rather than fail when the cached tokenizer is absent, so the suite
// stays runnable on a machine that never downloaded the checkpoint.
func loadTokB(b *testing.B) *tokenizer.Tokenizer {
	b.Helper()
	if p := lookupTokenizerPath(); p != "" {
		tk, err := tokenizer.Load(p)
		if err != nil {
			b.Fatalf("load tokenizer: %v", err)
		}
		return tk
	}
	b.Skip("laya tokenizer.json not found")
	return nil
}

// TestBuildWithStateMatchesBuild is the equivalence guard for the split between
// EncodeState and BuildWithState.
//
// The refactor exists so a batch of questions over one state encodes that state
// once instead of once per question. That is only safe if the two paths produce
// byte-identical sequences and marker positions, so this test pins them
// together across the cases that exercise each branch: no truncation, exact
// truncation, and a state far longer than the budget.
func TestBuildWithStateMatchesBuild(t *testing.T) {
	tk := loadTok(t)

	states := map[string]string{
		"empty":         "",
		"short":         `{"body":"We were billed twice for March."}`,
		"exact-ish":     strings.Repeat("word ", 100),
		"long":          strings.Repeat("This is a long document. ", 400),
		"mask-literal":  "the [MASK] token in a state must be neutralised",
		"unicode":       "日本語のテキストと Ünïcödé と 🚀 emoji",
		"whitespace":    "  leading and trailing  \n\ttabs\r\n",
	}

	questions := map[string]Question{
		"choice3": {
			Type:         Choice,
			Instructions: "Which department should handle this?",
			Criteria:     []string{"billing: invoices", "technical: bugs", "sales: pricing"},
			Labels:       []string{"billing", "technical", "sales"},
		},
		"choice1": {
			Type:         Choice,
			Instructions: "Is this spam?",
			Criteria:     []string{"yes"},
			Labels:       []string{"yes"},
		},
		"score": {
			Type:         Score,
			Instructions: "How urgent is this request?",
			Criteria:     []string{"level 0: not urgent", "level 1: soon", "level 2: critical"},
		},
		"noul": {
			Type:         NoUL,
			Instructions: "Does the user threaten to cancel or leave?",
		},
		"masky": {
			Type:         Choice,
			Instructions: "Which [MASK] option applies?",
			Criteria:     []string{"a: has [MASK] inside", "b: plain"},
			Labels:       []string{"a", "b"},
		},
	}

	budgets := []struct{ maxLen, headMaxLen int }{
		{512, 192},
		{256, 128},
		{64, 48},
		{1024, 256},
	}

	for sName, state := range states {
		for qName, q := range questions {
			for _, b := range budgets {
				want, wantErr := Build(tk, state, q, b.maxLen, b.headMaxLen)
				got, gotErr := BuildWithState(tk, EncodeState(tk, state, 0), q, b.maxLen, b.headMaxLen)

				if (wantErr == nil) != (gotErr == nil) {
					t.Errorf("state=%s q=%s budget=%d/%d: error mismatch: Build=%v BuildWithState=%v",
						sName, qName, b.maxLen, b.headMaxLen, wantErr, gotErr)
					continue
				}
				if wantErr != nil {
					continue
				}

				if !equalInts(want.IDs, got.IDs) {
					t.Errorf("state=%s q=%s budget=%d/%d: ids differ\n  Build:          %v\n  BuildWithState: %v",
						sName, qName, b.maxLen, b.headMaxLen, want.IDs, got.IDs)
				}
				if !equalInts(want.MarkerPositions, got.MarkerPositions) {
					t.Errorf("state=%s q=%s budget=%d/%d: markers differ: %v vs %v",
						sName, qName, b.maxLen, b.headMaxLen, want.MarkerPositions, got.MarkerPositions)
				}
				if want.Truncated != got.Truncated {
					t.Errorf("state=%s q=%s budget=%d/%d: Truncated = %v, want %v",
						sName, qName, b.maxLen, b.headMaxLen, got.Truncated, want.Truncated)
				}
				if !equalStrings(want.Options, got.Options) {
					t.Errorf("state=%s q=%s: options differ: %v vs %v", sName, qName, want.Options, got.Options)
				}
				if want.QType != got.QType {
					t.Errorf("state=%s q=%s: qtype = %v, want %v", sName, qName, got.QType, want.QType)
				}
			}
		}
	}
}

// TestEncodeStateNeutralisesMaskLiteral pins the one behaviour EncodeState owns
// beyond delegating to the tokenizer: a state is never a place where the [MASK]
// literal should be honoured as a marker.
func TestEncodeStateNeutralisesMaskLiteral(t *testing.T) {
	tk := loadTok(t)

	withMask := EncodeState(tk, "before [MASK] after", 0)
	neutralised := tk.Encode("before   after")

	if !equalInts(withMask, neutralised) {
		t.Errorf("EncodeState did not neutralise [MASK]\n  got:  %v\n  want: %v", withMask, neutralised)
	}
	for i, id := range withMask {
		if id == tk.MaskID() {
			t.Errorf("EncodeState kept a MASK id at index %d", i)
		}
	}
}

// TestEncodeStateBudgetMatchesFullPrefix is the correctness guard for bounding
// state encoding by the token budget.
//
// EncodeState stops early once it has `budget` tokens, which is only equivalent
// to encoding everything and truncating if stopping cannot change an earlier
// token. That holds because pre-tokens are BPE'd independently and visited in
// order, but it is the kind of assumption that silently corrupts prompts if it
// is wrong, so it is pinned against the unbounded path here.
func TestEncodeStateBudgetMatchesFullPrefix(t *testing.T) {
	tk := loadTok(t)

	states := []string{
		"",
		"a",
		"hello world",
		`{"body":"We were billed twice for March."}`,
		strings.Repeat("This is a long document. ", 400),
		strings.Repeat("  spaced   out  ", 200),
		"日本語のテキスト " + strings.Repeat("テキスト", 500),
		"emoji 🚀🔥 " + strings.Repeat("🔥", 300),
		"the [MASK] literal inside a state",
		strings.Repeat("word ", 5000),
		"MiXeD123CaSe " + strings.Repeat("Ab1 ", 400),
	}

	for _, budget := range []int{1, 2, 7, 16, 64, 128, 511, 512, 513, 1024, 4096} {
		for _, state := range states {
			full := EncodeState(tk, state, 0)
			bounded := EncodeState(tk, state, budget)

			want := full
			if len(want) > budget {
				want = want[:budget]
			}

			if !equalInts(want, bounded) {
				t.Errorf("budget=%d state=%.40q: bounded encoding differs from the full prefix\n"+
					"  full len=%d  want len=%d  got len=%d\n  want: %v\n  got:  %v",
					budget, state, len(full), len(want), len(bounded), want, bounded)
				continue
			}
			if len(bounded) > budget {
				t.Errorf("budget=%d state=%.40q: produced %d tokens, over budget",
					budget, state, len(bounded))
			}
		}
	}
}

// TestBuildMatchesUnderBoundedStateEncoding proves the budget is safe at the
// level that matters: the sequence and markers a caller actually sees.
//
// A question can consume at most maxLen state tokens (the layout adds at least
// one SEP on each side), so bounding the state at maxLen must not change the
// built sequence for any state, however long.
func TestBuildMatchesUnderBoundedStateEncoding(t *testing.T) {
	tk := loadTok(t)

	states := []string{
		"",
		"short",
		strings.Repeat("This is a long document. ", 400),
		strings.Repeat("word ", 5000),
		strings.Repeat("日本語テキスト", 800),
	}
	q := Question{
		Type:         Choice,
		Instructions: "Which department should handle this request?",
		Criteria:     []string{"billing: invoices", "technical: bugs", "sales: pricing"},
		Labels:       []string{"billing", "technical", "sales"},
	}

	for _, b := range []struct{ maxLen, headMaxLen int }{{512, 192}, {256, 128}, {1024, 256}, {64, 48}} {
		for _, state := range states {
			// Unbounded: encode everything, as the old per-question path did.
			unbounded := EncodeState(tk, state, 0)
			want, err := BuildWithState(tk, unbounded, q, b.maxLen, b.headMaxLen)
			if err != nil {
				t.Fatalf("BuildWithState(unbounded): %v", err)
			}

			// Bounded by maxLen: what the service now passes.
			boundedIDs := EncodeState(tk, state, b.maxLen)
			got, err := BuildWithState(tk, boundedIDs, q, b.maxLen, b.headMaxLen)
			if err != nil {
				t.Fatalf("BuildWithState(bounded): %v", err)
			}

			if !equalInts(want.IDs, got.IDs) {
				t.Errorf("maxLen=%d state=%.40q: ids differ between bounded and unbounded encoding\n"+
					"  want len=%d\n  got  len=%d", b.maxLen, state, len(want.IDs), len(got.IDs))
			}
			if !equalInts(want.MarkerPositions, got.MarkerPositions) {
				t.Errorf("maxLen=%d state=%.40q: markers differ: %v vs %v",
					b.maxLen, state, want.MarkerPositions, got.MarkerPositions)
			}
			if want.Truncated != got.Truncated {
				t.Errorf("maxLen=%d state=%.40q: Truncated = %v, want %v",
					b.maxLen, state, got.Truncated, want.Truncated)
			}
		}
	}
}

func equalStrings(a, b []string) bool {
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

// benchState is a document long enough that the state, not the question head,
// dominates tokenisation — the regime the shared-encode path targets.
func benchState(repeats int) string {
	return strings.Repeat("We were billed twice for March. Please refund the duplicate today. ", repeats)
}

var benchQuestion = Question{
	Type:         Choice,
	Instructions: "Which department should handle this request?",
	Criteria:     []string{"billing: invoices, payments, refunds", "technical: bugs, outages", "sales: pricing"},
	Labels:       []string{"billing", "technical", "sales"},
}

// BenchmarkBuildPerQuestion reproduces the old shape: one Build per question,
// each re-encoding the whole state.
func BenchmarkBuildPerQuestion(b *testing.B) {
	tk := loadTokB(b)
	state := benchState(120)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for q := 0; q < 10; q++ {
			if _, err := Build(tk, state, benchQuestion, 512, 256); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkBuildSharedState is the new shape: encode once, build ten questions.
func BenchmarkBuildSharedState(b *testing.B) {
	tk := loadTokB(b)
	state := benchState(120)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := EncodeState(tk, state, 512)
		for q := 0; q < 10; q++ {
			if _, err := BuildWithState(tk, ids, benchQuestion, 512, 256); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkEncodeState measures the unbounded cost on a long document: the
// whole state is tokenised even though the layout can only keep its head.
func BenchmarkEncodeState(b *testing.B) {
	tk := loadTokB(b)
	state := benchState(120)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EncodeState(tk, state, 0)
	}
}

// BenchmarkEncodeStateBounded measures the same state with the budget the
// service actually passes, so the saving from stopping early is visible.
func BenchmarkEncodeStateBounded(b *testing.B) {
	tk := loadTokB(b)
	state := benchState(120)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EncodeState(tk, state, 512)
	}
}
