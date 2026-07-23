package tokenizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWordPiece_Encode(t *testing.T) {
	tok := ExampleWordPieceVocab()

	tests := []struct {
		name    string
		text    string
		wantLen int
	}{
		{
			name:    "whole word in vocab",
			text:    "hello",
			wantLen: 1, // "hello" matches directly (greedy longest match).
		},
		{
			name:    "empty string",
			text:    "",
			wantLen: 0,
		},
		{
			name:    "two whole words",
			text:    "hello world",
			wantLen: 2, // "hello", "world" - both whole-word vocab hits.
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tok.Encode(tt.text)
			require.NoError(t, err)
			assert.Equal(t, tt.wantLen, len(tokens))
		})
	}
}

func TestWordPiece_Encode_SubwordSplit(t *testing.T) {
	// Vocab deliberately has no whole-word entry for "hello" so greedy
	// matching must fall back to "he" + "##llo".
	vocab := map[string]int32{
		"[UNK]": 0,
		"he":    1,
		"##llo": 2,
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(-1, -1, -1, 0)

	tokens, err := tok.Encode("hello")
	require.NoError(t, err)
	assert.Equal(t, []int32{1, 2}, tokens)
}

func TestWordPiece_Encode_UnknownWordFallsBackToUNK(t *testing.T) {
	vocab := map[string]int32{
		"[UNK]": 0,
		"he":    1,
		"##llo": 2,
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(-1, -1, -1, 0)

	// No valid subword split exists for "xyz" in this vocab.
	tokens, err := tok.Encode("xyz")
	require.NoError(t, err)
	assert.Equal(t, []int32{0}, tokens)
}

func TestWordPiece_Decode(t *testing.T) {
	tok := ExampleWordPieceVocab()

	tests := []struct {
		name   string
		tokens []int32
	}{
		{
			name:   "simple tokens",
			tokens: []int32{4, 5},
		},
		{
			name:   "empty tokens",
			tokens: []int32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, err := tok.Decode(tt.tokens)
			require.NoError(t, err)
			assert.NotNil(t, text)
		})
	}
}

func TestWordPiece_DecodeStripsContinuationPrefix(t *testing.T) {
	vocab := map[string]int32{
		"[UNK]": 0,
		"he":    1,
		"##llo": 2,
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(-1, -1, -1, 0)

	text, err := tok.Decode([]int32{1, 2})
	require.NoError(t, err)
	assert.Equal(t, "hello", text)
}

func TestWordPiece_VocabSize(t *testing.T) {
	tok := ExampleWordPieceVocab()
	assert.Greater(t, tok.VocabSize(), 0)
}

func TestWordPiece_SpecialTokens(t *testing.T) {
	vocab := map[string]int32{
		"[CLS]": 0,
		"[SEP]": 1,
		"[PAD]": 2,
		"[UNK]": 3,
		"a":     4,
		"b":     5,
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(0, 1, 2, 3)

	t.Run("bos token", func(t *testing.T) {
		assert.Equal(t, int32(0), tok.BosToken())
		assert.True(t, tok.IsSpecialToken(0))
	})

	t.Run("eos token", func(t *testing.T) {
		assert.Equal(t, int32(1), tok.EosToken())
		assert.True(t, tok.IsSpecialToken(1))
	})

	t.Run("pad token", func(t *testing.T) {
		assert.Equal(t, int32(2), tok.PadToken())
		assert.True(t, tok.IsSpecialToken(2))
	})

	t.Run("unk token", func(t *testing.T) {
		assert.Equal(t, int32(3), tok.UnkToken())
		assert.True(t, tok.IsSpecialToken(3))
	})

	t.Run("regular token", func(t *testing.T) {
		assert.False(t, tok.IsSpecialToken(4))
		assert.False(t, tok.IsSpecialToken(5))
	})
}

func TestWordPiece_NewWordPieceTokenizer(t *testing.T) {
	vocab := map[string]int32{
		"a":   0,
		"b":   1,
		"##b": 2,
	}
	tok := NewWordPieceTokenizer(vocab)
	require.NotNil(t, tok)
	assert.Equal(t, 3, tok.VocabSize())
}

func TestWordPiece_SetSpecialTokens(t *testing.T) {
	tok := ExampleWordPieceVocab()

	// Initially no special tokens set on a freshly constructed tokenizer.
	fresh := NewWordPieceTokenizer(map[string]int32{"a": 0})
	assert.Equal(t, int32(-1), fresh.BosToken())
	assert.Equal(t, int32(-1), fresh.EosToken())

	// Set special tokens.
	tok.SetSpecialTokens(100, 101, 102, 103)
	assert.Equal(t, int32(100), tok.BosToken())
	assert.Equal(t, int32(101), tok.EosToken())
	assert.Equal(t, int32(102), tok.PadToken())
	assert.Equal(t, int32(103), tok.UnkToken())
	assert.True(t, tok.IsSpecialToken(100))
	assert.True(t, tok.IsSpecialToken(101))
	assert.True(t, tok.IsSpecialToken(102))
	assert.True(t, tok.IsSpecialToken(103))
}

func TestWordPiece_EmptyVocab(t *testing.T) {
	tok := NewWordPieceTokenizer(map[string]int32{})
	tokens, err := tok.Encode("test")
	require.NoError(t, err)
	// No UNK token configured (unkToken == -1), so unmatched words produce
	// no tokens at all.
	assert.Empty(t, tokens)
}

func TestWordPiece_DecodeUnknownToken(t *testing.T) {
	tok := ExampleWordPieceVocab()
	// Token ID that doesn't exist in vocab.
	text, err := tok.Decode([]int32{9999})
	require.NoError(t, err)
	// Should contain replacement character.
	assert.Contains(t, text, "�")
}

func TestWordPiece_EncodeWord(t *testing.T) {
	vocab := map[string]int32{
		"[UNK]": 0,
		"he":    1,
		"##llo": 2,
		"world": 3,
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(-1, -1, -1, 0)

	t.Run("successful subword split", func(t *testing.T) {
		tokens, ok := tok.encodeWord("hello")
		assert.True(t, ok)
		assert.Equal(t, []int32{1, 2}, tokens)
	})

	t.Run("whole word match", func(t *testing.T) {
		tokens, ok := tok.encodeWord("world")
		assert.True(t, ok)
		assert.Equal(t, []int32{3}, tokens)
	})

	t.Run("no valid split", func(t *testing.T) {
		tokens, ok := tok.encodeWord("xyz")
		assert.False(t, ok)
		assert.Nil(t, tokens)
	})

	t.Run("word exceeds max input chars", func(t *testing.T) {
		tok.maxInputCharsPerWord = 3
		defer func() { tok.maxInputCharsPerWord = defaultMaxInputCharsPerWord }()

		tokens, ok := tok.encodeWord("hello")
		assert.False(t, ok)
		assert.Nil(t, tokens)
	})
}

func TestWordPiece_ContinuingSubwordPrefixConfigurable(t *testing.T) {
	vocab := map[string]int32{
		"[UNK]": 0,
		"he":    1,
		"@@llo": 2, // non-default continuation prefix
	}
	tok := NewWordPieceTokenizer(vocab)
	tok.SetSpecialTokens(-1, -1, -1, 0)
	tok.continuingSubwordPre = "@@"

	tokens, err := tok.Encode("hello")
	require.NoError(t, err)
	assert.Equal(t, []int32{1, 2}, tokens)
}
