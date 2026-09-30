// Package tokenizer implements the ByteLevel BPE tokenizer the laya
// checkpoints ship, matching the Python `tokenizers` library's output for the
// same tokenizer.json.
//
// Why this exists rather than calling out to Python: the whole point of running
// the model through TensorRT is to keep the per-request path in one process.
// Shelling out for tokenisation would dominate the 4 ms the kernel takes.
//
// The pipeline mirrors tokenizer.json exactly:
//
//	normalizer     NFC
//	pre_tokenizer  ByteLevel(add_prefix_space=false, use_regex=true)
//	model          BPE, 50280 base vocab + 116 added tokens
//	post_processor TemplateProcessing: [CLS] A [SEP]
//
// The byte-level pre-tokenizer maps every byte to a printable "unicode char"
// via the GPT-2 byte↔unicode table, then splits on the GPT-2 regex, then BPE
// merges within each split. Getting this wrong shifts every token id, so
// TestParityAgainstPython checks it against the Python tokenizer directly.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// SpecialTokenIDs for the laya checkpoints. Read from the file at load time;
// these are the fallback if a tokenizer.json omits the post-processor.
const (
	DefaultCLSID  = 50281
	DefaultSEPID  = 50282
	DefaultPadID  = 50283
	DefaultMaskID = 50284
	DefaultUnkID  = 50280
)

// Tokenizer is a loaded ByteLevel BPE vocabulary. Safe for concurrent use.
type Tokenizer struct {
	vocab     map[string]int
	idsToToks []string
	ranks     map[string]int // "a b" -> merge rank
	added     []addedToken

	clsID, sepID, padID, maskID, unkID int

	// cache memoises BPE results per pre-token, which is where the time goes.
	cache sync.Map

	addedRE *regexp.Regexp

	// addedByContent maps an added token's text to its id. addedID used to scan
	// all 116 entries per match.
	addedByContent map[string]int

	// Fast-path index for splitOnAdded. A typical document contains none of the
	// added tokens, yet the 116-alternative regex was measured at 90% of encode
	// time. These fields let one byte scan decide the common case first.
	//
	// addedFirstByte marks the first byte of every added token that is not an
	// all-space run; space runs are handled by minSpaceRun instead, because a
	// single space is far too common to use as a trigger.
	addedFirstByte [256]bool
	minSpaceRun    int
}

type addedToken struct {
	content string
	id      int
	special bool
	lstrip  bool
	rstrip  bool
	singleW bool
	normz   bool
}

// tokenizerJSON is the subset of tokenizer.json this implementation reads.
type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
		Lstrip  bool   `json:"lstrip"`
		Rstrip  bool   `json:"rstrip"`
		SingleW bool   `json:"single_word"`
		Normz   bool   `json:"normalized"`
	} `json:"added_tokens"`
	Model struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`
	PostProcessor struct {
		Type    string `json:"type"`
		Special map[string]struct {
			IDs []int `json:"ids"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

// Load reads a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", path, err)
	}
	return Parse(raw)
}

// Parse builds a tokenizer from tokenizer.json bytes.
func Parse(raw []byte) (*Tokenizer, error) {
	var doc tokenizerJSON
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("tokenizer: parse: %w", err)
	}
	if doc.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (only BPE)", doc.Model.Type)
	}
	if len(doc.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: vocabulary is empty")
	}

	t := &Tokenizer{
		vocab:  doc.Model.Vocab,
		ranks:  make(map[string]int, len(doc.Model.Vocab)),
		clsID:  DefaultCLSID,
		sepID:  DefaultSEPID,
		padID:  DefaultPadID,
		maskID: DefaultMaskID,
		unkID:  DefaultUnkID,
	}

	// merges is either [["a","b"],...] (current) or ["a b",...] (older).
	if err := t.parseMerges(doc.Model.Merges); err != nil {
		return nil, err
	}

	t.idsToToks = make([]string, len(t.vocab))
	for tok, id := range t.vocab {
		if id >= 0 && id < len(t.idsToToks) {
			t.idsToToks[id] = tok
		}
	}

	for _, at := range doc.AddedTokens {
		t.added = append(t.added, addedToken{
			content: at.Content, id: at.ID, special: at.Special,
			lstrip: at.Lstrip, rstrip: at.Rstrip, singleW: at.SingleW, normz: at.Normz,
		})
		// An added token occupies its id and is matched before BPE.
		if at.ID >= len(t.idsToToks) {
			grown := make([]string, at.ID+1)
			copy(grown, t.idsToToks)
			t.idsToToks = grown
		}
		if at.Special {
			t.idsToToks[at.ID] = at.Content
		}
	}

	// Post-processor special ids, when the file provides them.
	if ids := doc.PostProcessor.Special["[CLS]"].IDs; len(ids) == 1 {
		t.clsID = ids[0]
	}
	if ids := doc.PostProcessor.Special["[SEP]"].IDs; len(ids) == 1 {
		t.sepID = ids[0]
	}
	if ids := doc.PostProcessor.Special["[PAD]"].IDs; len(ids) == 1 {
		t.padID = ids[0]
	}
	if ids := doc.PostProcessor.Special["[MASK]"].IDs; len(ids) == 1 {
		t.maskID = ids[0]
	}
	if ids := doc.PostProcessor.Special["[UNK]"].IDs; len(ids) == 1 {
		t.unkID = ids[0]
	}

	// Added tokens are matched as whole strings, longest first, before the
	// GPT-2 rule runs. Both special and ordinary entries participate: the
	// checkpoint registers space runs and markers as real vocabulary entries,
	// and the reference splits on them.
	var alternatives []string
	t.addedByContent = make(map[string]int, len(t.added))
	t.minSpaceRun = 0
	for _, at := range t.added {
		if at.content == "" {
			continue
		}
		alternatives = append(alternatives, regexp.QuoteMeta(at.content))
		if _, seen := t.addedByContent[at.content]; !seen {
			t.addedByContent[at.content] = at.id
		}

		// Index the token for the split fast path.
		if isAllSpaces(at.content) {
			// A run of n spaces can only occur inside a run of >= n spaces, so
			// the smallest such n is the only threshold worth testing.
			if n := len(at.content); t.minSpaceRun == 0 || n < t.minSpaceRun {
				t.minSpaceRun = n
			}
			continue
		}
		t.addedFirstByte[at.content[0]] = true
	}
	if len(alternatives) > 0 {
		sort.Slice(alternatives, func(i, j int) bool {
			return len(alternatives[i]) > len(alternatives[j])
		})
		t.addedRE = regexp.MustCompile("(" + strings.Join(alternatives, "|") + ")")
	}

	return t, nil
}

// isAllSpaces reports whether s is a non-empty run of ASCII spaces, which is
// how the checkpoints register 2..24 space runs.
func isAllSpaces(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' {
			return false
		}
	}
	return len(s) > 0
}

// mayContainAdded reports whether text could contain any added token.
//
// This is a conservative over-approximation: it returns true whenever a match is
// possible, and only returns false when no added token can occur. The regex is
// then skipped for the common case of ordinary prose, which contains none of the
// 116 entries (88 are [unusedN], 23 are space runs of 2..24).
//
// Two triggers are enough to be exact:
//
//   - any byte that starts a non-space added token (for the checkpoint that is
//     '[', '|', '<')
//   - a run of at least minSpaceRun spaces
//
// A token consisting only of spaces cannot start anywhere else, because a run of
// n spaces only exists inside a run of at least n spaces.
func (t *Tokenizer) mayContainAdded(text string) bool {
	if t.addedRE == nil {
		return false
	}
	run := 0
	for i := 0; i < len(text); i++ {
		c := text[i]
		if t.addedFirstByte[c] {
			return true
		}
		if c == ' ' {
			run++
			if t.minSpaceRun > 0 && run >= t.minSpaceRun {
				return true
			}
			continue
		}
		run = 0
	}
	return false
}

func (t *Tokenizer) parseMerges(raw json.RawMessage) error {
	if len(raw) == 0 {
		return fmt.Errorf("tokenizer: model.merges is missing")
	}
	// Try the pair form first.
	var pairs [][]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		for i, p := range pairs {
			if len(p) != 2 {
				return fmt.Errorf("tokenizer: merge %d is not a pair", i)
			}
			t.ranks[p[0]+" "+p[1]] = i
		}
		return nil
	}
	// Fall back to the space-separated string form.
	var lines []string
	if err := json.Unmarshal(raw, &lines); err != nil {
		return fmt.Errorf("tokenizer: model.merges has an unsupported shape: %w", err)
	}
	for i, line := range lines {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		t.ranks[parts[0]+" "+parts[1]] = i
	}
	return nil
}

// Accessors for the ids callers need when building sequences.

func (t *Tokenizer) CLSID() int  { return t.clsID }
func (t *Tokenizer) SEPID() int  { return t.sepID }
func (t *Tokenizer) PadID() int  { return t.padID }
func (t *Tokenizer) MaskID() int { return t.maskID }
func (t *Tokenizer) UnkID() int  { return t.unkID }

// VocabSize is the number of entries in the vocabulary table.
func (t *Tokenizer) VocabSize() int { return len(t.idsToToks) }

// Encode tokenises text without special tokens.
func (t *Tokenizer) Encode(text string) []int {
	return t.encode(text, false)
}

// EncodeWithSpecials wraps the result in [CLS] ... [SEP], as the post-processor
// in tokenizer.json specifies for a single sequence.
func (t *Tokenizer) EncodeWithSpecials(text string) []int {
	return t.encode(text, true)
}

func (t *Tokenizer) encode(text string, withSpecials bool) []int {
	ids := t.encodeNoSpecials(text)
	if !withSpecials {
		return ids
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, t.clsID)
	out = append(out, ids...)
	out = append(out, t.sepID)
	return out
}

// EncodePrefix tokenises text and returns at most maxTokens ids, stopping as
// soon as that many are available.
//
// When the full encoding is longer, the result is exactly Encode(text)[:maxTokens].
// That holds because pre-tokens are BPE'd independently and are visited in the
// same order, so stopping early cannot change an earlier token.
//
// This exists so a long state is not fully tokenised only to be truncated to a
// fraction of its length: the caller bounds the work by the budget it can
// actually use. Sequence layout truncates the state from the right, keeping the
// head, so the dropped tail never influences the result.
func (t *Tokenizer) EncodePrefix(text string, maxTokens int) []int {
	if maxTokens <= 0 {
		return nil
	}
	text = norm.NFC.String(text)

	ids := make([]int, 0, maxTokens)
	for _, segment := range t.splitOnAdded(text) {
		if segment.id >= 0 {
			ids = append(ids, segment.id)
			if len(ids) >= maxTokens {
				break
			}
			continue
		}
		if segment.text == "" {
			continue
		}
		for _, piece := range gpt2Split([]rune(segment.text)) {
			ids = append(ids, t.bpe(byteLevelMap(piece))...)
			if len(ids) >= maxTokens {
				break
			}
		}
		if len(ids) >= maxTokens {
			break
		}
	}
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	return ids
}

// encodeNoSpecials is the raw pipeline, matching the reference order exactly:
//
//  1. NFC normalise
//  2. split on every added token (special AND ordinary, longest match first)
//  3. apply the GPT-2 split rule to the ORIGINAL text of each remaining run
//  4. byte-map each pre-token into the ByteLevel alphabet
//  5. BPE-merge each pre-token
//
// Step 3 before step 4 is the part that is easy to get backwards. The regex is
// matched against the real characters, so " 3" is one pre-token (space+digits)
// and only then becomes "Ġ3"; splitting the mapped text instead would cut
// between "Ġ" and "3" and produce different ids.
func (t *Tokenizer) encodeNoSpecials(text string) []int {
	text = norm.NFC.String(text)

	var ids []int
	for _, segment := range t.splitOnAdded(text) {
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

type segment struct {
	text string
	id   int
}

// splitOnAdded cuts the text at every added-token boundary. Ordinary added
// tokens count too: the checkpoint registers runs of 2..24 spaces as real
// vocabulary entries, and the reference emits them as single ids.
func (t *Tokenizer) splitOnAdded(text string) []segment {
	if t.addedRE == nil {
		return []segment{{text: text, id: -1}}
	}
	// One byte scan decides the overwhelmingly common case — ordinary text with
	// no added token in it — before the 116-alternative regex runs. This is the
	// single hottest operation in tokenisation.
	//
	// Empty input yields no segments, matching the regex path: Split("") is
	// [""], and the empty part is dropped.
	if text == "" {
		return nil
	}
	if !t.mayContainAdded(text) {
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

func (t *Tokenizer) addedID(content string) int {
	if id, ok := t.addedByContent[content]; ok {
		return id
	}
	return -1
}

// bpe applies the merge rules to one pre-token and maps the result to ids.
func (t *Tokenizer) bpe(piece string) []int {
	if piece == "" {
		return nil
	}
	if cached, ok := t.cache.Load(piece); ok {
		return cached.([]int)
	}

	symbols := make([]string, 0, len(piece))
	for _, r := range piece {
		symbols = append(symbols, string(r))
	}

	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(symbols); i++ {
			rank, ok := t.ranks[symbols[i]+" "+symbols[i+1]]
			if !ok {
				continue
			}
			if bestRank == -1 || rank < bestRank {
				bestRank, bestIdx = rank, i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx], append([]string{merged}, symbols[bestIdx+2:]...)...)
	}

	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		} else {
			// A symbol absent from the vocab maps to [UNK] per character, which
			// is what the reference tokenizer does when no unk_token is set:
			// the piece is emitted as its own byte-level fallback.
			ids = append(ids, t.unkID)
		}
	}

	t.cache.Store(piece, ids)
	return ids
}

// Decode maps ids back to text, for debugging and the /tokenize endpoint.
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.idsToToks) {
			continue
		}
		tok := t.idsToToks[id]
		if tok == "" {
			continue
		}
		sb.WriteString(byteLevelDecode(tok))
	}
	return sb.String()
}

// Tokens returns the vocabulary strings for ids, for the /tokenize endpoint.
func (t *Tokenizer) Tokens(ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		if id >= 0 && id < len(t.idsToToks) {
			out[i] = t.idsToToks[id]
		}
	}
	return out
}

// IsSpecial reports whether an id is one of the checkpoint's special tokens.
func (t *Tokenizer) IsSpecial(id int) bool {
	switch id {
	case t.clsID, t.sepID, t.padID, t.maskID:
		return true
	}
	return false
}

// ── ByteLevel ───────────────────────────────────────────────────────────────

// gpt2ByteToRune is the byte→unicode table from the GPT-2/ByteLevel
// pre-tokenizer. Bytes that are already printable ASCII map to themselves; the
// rest are shifted into a private range so every byte becomes a visible rune.
var (
	gpt2ByteToRune [256]rune
	gpt2RuneToByte = map[rune]byte{}
	byteLevelOnce  sync.Once
)

func initByteLevel() {
	byteLevelOnce.Do(func() {
		var bs []int
		for b := '!'; b <= '~'; b++ {
			bs = append(bs, int(b))
		}
		for b := 0xA1; b <= 0xAC; b++ {
			bs = append(bs, int(b))
		}
		for b := 0xAE; b <= 0xFF; b++ {
			bs = append(bs, int(b))
		}
		inSet := func(b int) bool {
			for _, x := range bs {
				if x == b {
					return true
				}
			}
			return false
		}
		n := 0
		for b := 0; b < 256; b++ {
			if inSet(b) {
				gpt2ByteToRune[b] = rune(b)
			} else {
				gpt2ByteToRune[b] = rune(256 + n)
				n++
			}
			gpt2RuneToByte[gpt2ByteToRune[b]] = byte(b)
		}
	})
}

// The GPT-2 pre-tokenizer pattern, for reference:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Go's regexp is RE2 and rejects the `(?!\S)` lookahead, so the split is done
// by hand below. The alternation order is significant and is preserved.

var contractions = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// isSpaceRune matches Go's regexp `\s` class exactly: [\t\n\f\r ].
func isSpaceRune(r rune) bool {
	switch r {
	case '\t', '\n', '\f', '\r', ' ':
		return true
	}
	return false
}

// byteLevelMap maps text into the ByteLevel alphabet: every input byte becomes
// one printable rune via the GPT-2 table.
func byteLevelMap(text string) string {
	initByteLevel()
	var sb strings.Builder
	sb.Grow(len(text))
	for i := 0; i < len(text); i++ {
		sb.WriteRune(gpt2ByteToRune[text[i]])
	}
	return sb.String()
}

func gpt2Split(rs []rune) []string {
	n := len(rs)
	if n == 0 {
		return nil
	}
	out := make([]string, 0, n/2+1)

	for i := 0; i < n; {
		start := i

		// 1. contractions: 's 't 're 've 'm 'll 'd
		if rs[i] == '\'' {
			for _, c := range contractions {
				cr := []rune(c)
				if i+len(cr) <= n && string(rs[i:i+len(cr)]) == c {
					i += len(cr)
					break
				}
			}
			if i != start {
				out = append(out, string(rs[start:i]))
				continue
			}
		}

		// 2. " ?\p{L}+"
		if j, ok := scanWithOptionalSpace(rs, i, unicode.IsLetter); ok {
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// 3. " ?\p{N}+"
		if j, ok := scanWithOptionalSpace(rs, i, unicode.IsNumber); ok {
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// 4. " ?[^\s\p{L}\p{N}]+"
		if j, ok := scanWithOptionalSpace(rs, i, func(r rune) bool {
			return !isSpaceRune(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
		}); ok {
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// 5/6. "\s+(?!\S)" then "\s+". Greedy whitespace, but the run gives
		// back its final rune when a non-space follows, so that rune stays with
		// the word the next iteration will match ("  hello" -> " ", " hello").
		if isSpaceRune(rs[i]) {
			j := i
			for j < n && isSpaceRune(rs[j]) {
				j++
			}
			if j < n && j-i > 1 {
				j--
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// Should be unreachable; consume one rune so the loop always advances.
		i++
		out = append(out, string(rs[start:i]))
	}
	return out
}

// scanWithOptionalSpace matches an optional single space followed by one or
// more runes satisfying pred. It returns the end index and whether it matched.
func scanWithOptionalSpace(rs []rune, i int, pred func(rune) bool) (int, bool) {
	n := len(rs)
	j := i
	if j < n && rs[j] == ' ' {
		j++
	}
	if j >= n || !pred(rs[j]) {
		return i, false
	}
	for j < n && pred(rs[j]) {
		j++
	}
	return j, true
}

// byteLevelDecode reverses the mapping for one token string.
func byteLevelDecode(tok string) string {
	initByteLevel()
	var buf []byte
	for _, r := range tok {
		if b, ok := gpt2RuneToByte[r]; ok {
			buf = append(buf, b)
			continue
		}
		// Not part of the byte alphabet: emit it as-is (UTF-8 encoded).
		var tmp [utf8.UTFMax]byte
		n := utf8.EncodeRune(tmp[:], r)
		buf = append(buf, tmp[:n]...)
	}
	return string(buf)
}
