namespace wtpsplit;
using System.Collections.Generic;

/// <summary>Defines the interface for a tokenizer used in SaT.</summary>
public interface ITokenizer {

    /// <summary>Token ID representing the beginning-of-sequence (BOS) token.</summary>
    int BosToken { get; }

    /// <summary>Token ID representing the end-of-sequence (EOS) token.</summary>
    int EosToken { get; }

    /// <summary>Token ID used for padding.</summary>
    int PadToken { get; }

    /// <summary>Encodes a string into a sequence of tokens with character offsets.</summary>
    /// <param name="text">The input text to tokenize.</param>
    /// <returns>List of tokens with offset mappings.</returns>
    IReadOnlyList<Token> Encode(string text);

    /// <summary>Represents a single token with its ID and character offset in the original text.</summary>
    /// <param name="TokenId">The token ID.</param>
    /// <param name="Offset">The character offset range in the input text.</param>
    public record struct Token(int TokenId, Offset Offset);

    /// <summary>Represents the start and end character positions of a token in the input text.</summary>
    /// <param name="Start">Inclusive start index of the token.</param>
    /// <param name="End">Exclusive end index of the token.</param>
    public record struct Offset(int Start, int End);
}