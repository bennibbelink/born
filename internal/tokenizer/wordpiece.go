package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// WordPieceTokenizer implements the WordPiece tokenization algorithm used by
// BERT-style models (e.g. BGE, GTE, E5, MiniLM).
//
// This is a pure Go implementation that can load HuggingFace tokenizer.json
// files whose "model" section has "type": "WordPiece".
type WordPieceTokenizer struct {
	vocab                map[string]int32 // token -> ID
	reverseVocab         map[int32]string // ID -> token
	normalizer           Normalizer       // text normalizer (from tokenizer.json)
	continuingSubwordPre string           // prefix for non-initial subword pieces, e.g. "##"
	maxInputCharsPerWord int              // words longer than this become [UNK]
	bosToken             int32
	eosToken             int32
	padToken             int32
	unkToken             int32
	unkTokenStr          string
	specialTokens        map[int32]bool
}

const defaultMaxInputCharsPerWord = 100

// NewWordPieceTokenizer creates a new WordPiece tokenizer from a vocab map.
func NewWordPieceTokenizer(vocab map[string]int32) *WordPieceTokenizer {
	reverseVocab := make(map[int32]string, len(vocab))
	for token, id := range vocab {
		reverseVocab[id] = token
	}

	return &WordPieceTokenizer{
		vocab:                vocab,
		reverseVocab:         reverseVocab,
		continuingSubwordPre: "##",
		maxInputCharsPerWord: defaultMaxInputCharsPerWord,
		bosToken:             -1,
		eosToken:             -1,
		padToken:             -1,
		unkToken:             -1,
		unkTokenStr:          "[UNK]",
		specialTokens:        make(map[int32]bool),
	}
}

// SetSpecialTokens configures special token IDs.
func (w *WordPieceTokenizer) SetSpecialTokens(bos, eos, pad, unk int32) {
	w.bosToken = bos
	w.eosToken = eos
	w.padToken = pad
	w.unkToken = unk

	if bos >= 0 {
		w.specialTokens[bos] = true
	}
	if eos >= 0 {
		w.specialTokens[eos] = true
	}
	if pad >= 0 {
		w.specialTokens[pad] = true
	}
	if unk >= 0 {
		w.specialTokens[unk] = true
		if text, ok := w.reverseVocab[unk]; ok {
			w.unkTokenStr = text
		}
	}
}

// Encode converts text to token IDs using WordPiece's greedy longest-match-first
// algorithm.
func (w *WordPieceTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return []int32{}, nil
	}

	if w.normalizer != nil {
		text = w.normalizer.Normalize(text)
	}

	words := basicSplit(text)
	var tokens []int32

	for _, word := range words {
		wordTokens, ok := w.encodeWord(word)
		if !ok {
			// Word couldn't be tokenized (too long or no valid split); fall back to [UNK].
			if w.unkToken >= 0 {
				tokens = append(tokens, w.unkToken)
			}
			continue
		}
		tokens = append(tokens, wordTokens...)
	}

	return tokens, nil
}

// encodeWord applies greedy longest-match-first subword matching to a single word.
// Returns (tokenIDs, true) on success, or (nil, false) if the word can't be
// represented (too long, or no valid subword split exists).
func (w *WordPieceTokenizer) encodeWord(word string) ([]int32, bool) {
	runes := []rune(word)
	if len(runes) > w.maxInputCharsPerWord {
		return nil, false
	}

	var tokenIDs []int32
	start := 0

	for start < len(runes) {
		end := len(runes)
		var matchedID int32
		matched := false

		// Try the longest possible substring first, shrinking until a match is found.
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = w.continuingSubwordPre + substr
			}

			if id, ok := w.vocab[substr]; ok {
				matchedID = id
				matched = true
				break
			}
			end--
		}

		if !matched {
			return nil, false
		}

		tokenIDs = append(tokenIDs, matchedID)
		start = end
	}

	return tokenIDs, true
}

// Decode converts token IDs back to text, stripping the continuation prefix
// and re-inserting spaces between whole words.
func (w *WordPieceTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder

	for i, token := range tokens {
		text, ok := w.reverseVocab[token]
		if !ok {
			text = "�"
		}

		isContinuation := strings.HasPrefix(text, w.continuingSubwordPre)
		if isContinuation {
			text = strings.TrimPrefix(text, w.continuingSubwordPre)
		} else if i > 0 {
			sb.WriteString(" ")
		}

		sb.WriteString(text)
	}

	return sb.String(), nil
}

// VocabSize returns the total vocabulary size.
func (w *WordPieceTokenizer) VocabSize() int {
	return len(w.vocab)
}

// BosToken returns the beginning-of-sequence token ID.
func (w *WordPieceTokenizer) BosToken() int32 {
	return w.bosToken
}

// EosToken returns the end-of-sequence token ID.
func (w *WordPieceTokenizer) EosToken() int32 {
	return w.eosToken
}

// PadToken returns the padding token ID.
func (w *WordPieceTokenizer) PadToken() int32 {
	return w.padToken
}

// UnkToken returns the unknown token ID.
func (w *WordPieceTokenizer) UnkToken() int32 {
	return w.unkToken
}

// IsSpecialToken checks if a token ID is a special token.
func (w *WordPieceTokenizer) IsSpecialToken(token int32) bool {
	return w.specialTokens[token]
}

// HuggingFaceWordPieceConfig represents the tokenizer.json structure for a
// WordPiece model.
type HuggingFaceWordPieceConfig struct {
	Model struct {
		Type                    string         `json:"type"`
		Vocab                   map[string]int `json:"vocab"`
		UnkToken                string         `json:"unk_token"`
		ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer json.RawMessage `json:"normalizer"`
}

// LoadWordPieceFromHuggingFace loads a WordPiece tokenizer from tokenizer.json.
//
// This is a simplified loader that handles the most common HuggingFace format
// (used by BERT-derived models such as BGE, GTE, E5, and MiniLM).
func LoadWordPieceFromHuggingFace(path string) (*WordPieceTokenizer, error) {
	data, err := os.ReadFile(path) //nolint:gosec // G304: Path comes from trusted caller
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json: %w", err)
	}

	var config HuggingFaceWordPieceConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer.json: %w", err)
	}

	vocab := make(map[string]int32, len(config.Model.Vocab))
	for token, id := range config.Model.Vocab {
		vocab[token] = int32(id) //nolint:gosec // G115: integer overflow conversion int -> int32
	}

	tokenizer := NewWordPieceTokenizer(vocab)

	if config.Model.ContinuingSubwordPrefix != "" {
		tokenizer.continuingSubwordPre = config.Model.ContinuingSubwordPrefix
	}
	if config.Model.MaxInputCharsPerWord > 0 {
		tokenizer.maxInputCharsPerWord = config.Model.MaxInputCharsPerWord
	}
	if config.Model.UnkToken != "" {
		tokenizer.unkTokenStr = config.Model.UnkToken
	}

	// Parse normalizer (Lowercase, StripAccents, Sequence, etc.).
	if len(config.Normalizer) > 0 {
		norm, err := parseNormalizer(config.Normalizer)
		if err != nil {
			return nil, fmt.Errorf("failed to parse normalizer: %w", err)
		}
		tokenizer.normalizer = norm
	}

	configureWordPieceSpecialTokens(tokenizer, config.AddedTokens)
	return tokenizer, nil
}

// configureWordPieceSpecialTokens identifies BOS/EOS/PAD/UNK from the
// added_tokens list, e.g. [CLS]/[SEP]/[PAD]/[UNK] for BERT-style vocabs.
func configureWordPieceSpecialTokens(tokenizer *WordPieceTokenizer, addedTokens []struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}) {
	for _, addedToken := range addedTokens {
		id := int32(addedToken.ID) //nolint:gosec // G115: integer overflow conversion int -> int32
		if !addedToken.Special {
			continue
		}

		tokenizer.specialTokens[id] = true

		content := strings.ToLower(addedToken.Content)
		switch {
		case content == "[cls]" || strings.Contains(content, "bos"):
			tokenizer.bosToken = id
		case content == "[sep]" || strings.Contains(content, "eos"):
			tokenizer.eosToken = id
		case strings.Contains(content, "pad"):
			tokenizer.padToken = id
		case content == "[unk]" || strings.Contains(content, "unk"):
			tokenizer.unkToken = id
			tokenizer.unkTokenStr = addedToken.Content
		}
	}
}

// basicSplit performs BERT-style basic tokenization: splitting on whitespace.
// A full BERT basic tokenizer also splits off punctuation into its own tokens;
// callers that need that behavior should pre-process text before calling
// Encode, or extend this function accordingly.
func basicSplit(text string) []string {
	return strings.Fields(text)
}

// ExampleWordPieceVocab creates a minimal WordPiece tokenizer for testing.
func ExampleWordPieceVocab() *WordPieceTokenizer {
	// Minimal vocab for demonstration: "hello world" -> "hello", "world".
	vocab := map[string]int32{
		"[UNK]": 0,
		"[CLS]": 1,
		"[SEP]": 2,
		"[PAD]": 3,
		"hello": 4,
		"world": 5,
		"wor":   6,
		"##ld":  7,
		"he":    8,
		"##llo": 9,
	}

	tokenizer := NewWordPieceTokenizer(vocab)
	tokenizer.SetSpecialTokens(1, 2, 3, 0)

	return tokenizer
}
