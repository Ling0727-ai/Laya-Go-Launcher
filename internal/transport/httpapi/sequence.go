package httpapi

import (
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/sequence"
)

// sequenceBuild renders a sequence through the service's tokenizer and budgets.
// It lives here rather than in the handler so the handler stays a decode/encode
// shim.
func sequenceBuild(s *Server, state any, q sequence.Question, maxLen, headMaxLen int) (sequence.Built, error) {
	return sequence.Build(s.inference.Tokenizer(), inference.SerializeState(state), q, maxLen, headMaxLen)
}
