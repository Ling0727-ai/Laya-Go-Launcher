package tokenizer

import (
	"strings"
	"testing"
)

// splitOnAddedReference is the pre-optimisation implementation: it always runs
// the 116-alternative regex. The fast path in splitOnAdded is only valid if it
// produces byte-identical output for every input, so it is compared against this
// directly rather than against a paraphrase of the same logic.
func (t *Tokenizer) splitOnAddedReference(text string) []segment {
	if t.addedRE == nil {
		return []segment{{text: text, id: -1}}
	}
	parts := t.addedRE.Split(text, -1)
	matches := t.addedRE.FindAllString(text, -1)

	out := make([]segment, 0, len(parts)+len(matches))
	for i, part := range parts {
		if part != "" {
			out = append(out, segment{text: part, id: -1})
		}
		if i < len(matches) {
			out = append(out, segment{text: matches[i], id: t.addedID(matches[i])})
		}
	}
	return out
}

func segmentsEqual(a, b []segment) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].text != b[i].text || a[i].id != b[i].id {
			return false
		}
	}
	return true
}

// addedTokenCorpus covers every shape that could distinguish the fast path from
// the regex: plain prose with no trigger, each non-space trigger character, space
// runs both below and above the minimum registered run, the added tokens
// themselves, and combinations at the string edges.
func addedTokenCorpus() []string {
	base := []string{
		"",
		"a",
		"hello world",
		"Duplicate charge on invoice #4411",
		"We were billed twice for March. Please refund the duplicate today.",
		"user@acme.com",
		"tab\tseparated\tvalues",
		"newline\nseparated",
		"Ünïcödé with diacritics: café, naïve, Zürich",
		"日本語のテキスト",
		"Русский текст",
		"العربية نص",
		"Emoji 🚀🔥 and symbols ©®™",
		"camelCaseIdentifier and snake_case_name",
		"Numbers 1234567890 and 3.14159",
		"punctuation!?,;:'\"()[]{}<>|/\\-_=+*&^%$#@~`",
		// triggers and near-triggers for non-space added tokens
		"[CLS]",
		"[MASK]",
		"[PAD]",
		"[SEP]",
		"[UNK]",
		"[unused0]",
		"[unused82]",
		"text with [MASK] inside",
		"[MASK] at the start",
		"at the end [MASK]",
		"[MASK][MASK][MASK]",
		"a[MASK]b",
		"<|endoftext|>",
		"<|padding|>",
		"prefix <|endoftext|> suffix",
		"|||EMAIL_ADDRESS|||",
		"|||IP_ADDRESS|||",
		"|||PHONE_NUMBER|||",
		"contact |||EMAIL_ADDRESS||| now",
		// a lone '[' or '|' or '<' is a trigger but not a match
		"[",
		"]",
		"|",
		"<",
		">",
		"just a [ bracket",
		"just a | pipe",
		"just a < angle",
		"[[[[[[[[[[",
		"||||||||||",
		"<<<<<<<<<<",
		// space runs: below, at, and above the minimum registered run (2)
		" ",
		"  ",
		"   ",
		"    ",
		"a b",
		"a  b",
		"a   b",
		"a    b",
		strings.Repeat(" ", 24),
		strings.Repeat(" ", 25),
		strings.Repeat(" ", 100),
		"  leading",
		"trailing  ",
		"  both  ",
		"a" + strings.Repeat(" ", 24) + "b",
		"a" + strings.Repeat(" ", 25) + "b",
		// interleaved
		"  [MASK]  ",
		" [CLS] " + strings.Repeat(" ", 10) + " [SEP] ",
		strings.Repeat("word  word ", 50),
		strings.Repeat("word   word ", 50),
		// long realistic documents
		strings.Repeat("This is a long document. ", 200),
		strings.Repeat("We were billed twice for March. ", 200),
		strings.Repeat("  indented line\n", 100),
	}

	// Exhaustive short strings over the trigger alphabet, which is where an
	// off-by-one in the byte scan would hide.
	alphabet := []byte{' ', '[', ']', '|', '<', '>', 'a', 'M', 'A', 'S', 'K'}
	for _, a := range alphabet {
		for _, b := range alphabet {
			for _, c := range alphabet {
				base = append(base, string([]byte{a, b, c}))
			}
		}
	}
	return base
}

// TestSplitOnAddedFastPathMatchesRegex is the equivalence guard for the
// added-token fast path.
func TestSplitOnAddedFastPathMatchesRegex(t *testing.T) {
	tk := loadTokenizer(t)

	checked, skipped := 0, 0
	for _, text := range addedTokenCorpus() {
		want := tk.splitOnAddedReference(text)
		got := tk.splitOnAdded(text)

		if !segmentsEqual(want, got) {
			t.Errorf("splitOnAdded disagrees for %.60q\n  regex: %v\n  fast:  %v", text, want, got)
			continue
		}
		checked++
		if !tk.mayContainAdded(text) {
			skipped++
		}
	}
	t.Logf("checked %d inputs; fast path skipped the regex on %d", checked, skipped)
	if skipped == 0 {
		t.Error("the fast path never triggered: the test is not exercising it")
	}
}

// TestMayContainAddedIsConservative pins the property the fast path relies on:
// when it reports false, the regex must genuinely find nothing. A false negative
// would silently drop an added token from the output.
func TestMayContainAddedIsConservative(t *testing.T) {
	tk := loadTokenizer(t)

	for _, text := range addedTokenCorpus() {
		if tk.mayContainAdded(text) {
			continue // reported "maybe": always safe
		}
		if matches := tk.addedRE.FindAllString(text, -1); len(matches) > 0 {
			t.Errorf("mayContainAdded returned false but the regex matched %v in %.60q",
				matches, text)
		}
	}
}

// TestEncodeFastPathMatchesReference checks the equivalence one level up: the
// ids produced for the whole corpus must not change.
func TestEncodeFastPathMatchesReference(t *testing.T) {
	tk := loadTokenizer(t)

	for _, text := range addedTokenCorpus() {
		got := tk.Encode(text)

		// Rebuild the encoding through the reference splitter so the comparison
		// exercises the same code path the old implementation used.
		ref := tk.encodeViaReferenceSplit(text)

		if !equalInts(ref, got) {
			t.Errorf("Encode disagrees for %.60q\n  reference: %v\n  fast:      %v", text, ref, got)
		}
	}
}

// encodeViaReferenceSplit is encodeNoSpecials with splitOnAddedReference
// substituted for the fast path.
func (t *Tokenizer) encodeViaReferenceSplit(text string) []int {
	text = nfcString(text)
	var ids []int
	for _, segment := range t.splitOnAddedReference(text) {
		if segment.id >= 0 {
			ids = append(ids, segment.id)
			continue
		}
		if segment.text == "" {
			continue
		}
		for _, piece := range gpt2Split([]rune(segment.text)) {
			ids = append(ids, t.bpe(byteLevelMap(piece))...)
		}
	}
	return ids
}

// TestAddedTokenLookupMatchesScan checks the map built at load time agrees with
// the linear scan it replaced.
func TestAddedTokenLookupMatchesScan(t *testing.T) {
	tk := loadTokenizer(t)

	for _, at := range tk.added {
		if at.content == "" {
			continue
		}
		want := -1
		for _, other := range tk.added {
			if other.content == at.content {
				want = other.id
				break
			}
		}
		if got := tk.addedID(at.content); got != want {
			t.Errorf("addedID(%q) = %d, want %d", at.content, got, want)
		}
	}

	// Unknown content must stay -1 rather than matching a prefix or empty key.
	for _, unknown := range []string{"", "[MASK]x", "x[MASK]", "[mask]", "|EMAIL|"} {
		if got := tk.addedID(unknown); got != -1 {
			t.Errorf("addedID(%q) = %d, want -1", unknown, got)
		}
	}
}

// TestMinSpaceRunMatchesRegisteredTokens verifies the space-run threshold comes
// from the vocabulary rather than a hardcoded guess.
func TestMinSpaceRunMatchesRegisteredTokens(t *testing.T) {
	tk := loadTokenizer(t)

	want := 0
	for _, at := range tk.added {
		if isAllSpaces(at.content) {
			if n := len(at.content); want == 0 || n < want {
				want = n
			}
		}
	}
	if tk.minSpaceRun != want {
		t.Errorf("minSpaceRun = %d, want %d", tk.minSpaceRun, want)
	}
	if want != 2 {
		t.Logf("note: this checkpoint's smallest space-run token is %d spaces", want)
	}
}
