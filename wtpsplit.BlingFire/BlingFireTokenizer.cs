namespace wtpsplit.BlingFire;

using System.Collections.Immutable;
using System.Text;

/// <summary>Abstract base class for tokenizers using the BlingFire library.</summary>
public abstract class BlingFireTokenizer : ITokenizer, IDisposable {
    private static readonly Decoder UTF8Decoder = Encoding.UTF8.GetDecoder();

    private int disposedValue = 1;
    private readonly int[] tempBuffer;

    /// <summary>Initializes the tokenizer with a raw model and max token buffer size.</summary>
    /// <param name="model">Byte array of the tokenizer model.</param>
    /// <param name="maxTokens">Maximum number of tokens the buffer can hold.</param>
    protected BlingFireTokenizer(byte[] model, int maxTokens) {
        tempBuffer = new int[maxTokens];
        Model = BlingFireUtils.SetModel(model, model.Length);
    }

    /// <summary>Initializes the tokenizer with a model file path and max token buffer size.</summary>
    /// <param name="path">File path to the tokenizer model.</param>
    /// <param name="maxTokens">Maximum number of tokens the buffer can hold.</param>
    protected BlingFireTokenizer(string path, int maxTokens) {
        tempBuffer = new int[maxTokens];
        Model = BlingFireUtils.LoadModel(Path.GetFullPath(path));
    }

    ~BlingFireTokenizer() {
        Dispose();
    }

    /// <summary>Releases the unmanaged model resource and suppresses finalization.</summary>
    public void Dispose() {
        if (Interlocked.Exchange(ref disposedValue, 0) == 1) {
            _ = BlingFireUtils.FreeModel(Model);
        }
        GC.SuppressFinalize(this);
    }

    private ulong Model { get; }

    /// <summary>Token ID used for unknown tokens.</summary>
    public abstract int UnkToken { get; }

    /// <summary>Token ID for beginning-of-sequence (BOS).</summary>
    public abstract int BosToken { get; }

    /// <summary>Token ID for end-of-sequence (EOS).</summary>
    public abstract int EosToken { get; }

    /// <summary>Token ID used for padding.</summary>
    public abstract int PadToken { get; }

    /// <summary>Tokenizes the input text and returns tokens with character offsets.</summary>
    /// <param name="text">The input string to tokenize.</param>
    /// <returns>List of tokens with offset mappings.</returns>
    public IReadOnlyList<ITokenizer.Token> Encode(string text) {
        // Convert the input text to a byte array using UTF-8 encoding
        byte[] buffer = Encoding.UTF8.GetBytes(text);

        // Ensure the buffer is large enough
        int count = GetTokenCount(buffer);
        int[] tokenIds = new int[count];
        int[] startOffsets = new int[count];
        int[] endOffsets = new int[count];

        // Tokenize the text and get token IDs with offsets
        _ = BlingFireUtils.TextToIdsWithOffsets(Model, buffer, buffer.Length, tokenIds, startOffsets, endOffsets, count, UnkToken);

        // Get char offsets from byte offsets
        IEnumerable<ITokenizer.Offset> offsets = startOffsets.Zip(endOffsets, (start, end) => {
            int sChar = UTF8Decoder.GetCharCount(buffer, 0, Math.Max(start, 0), false);
            int eChar = UTF8Decoder.GetCharCount(buffer, 0, end + 1, true);
            return new ITokenizer.Offset(sChar, eChar);
        });

        // Convert token IDs and offsets to the appropriate types
        return tokenIds.Zip(offsets, (id, offset) => new ITokenizer.Token(id, offset)).ToImmutableList();
    }

    private int GetTokenCount(byte[] buffer) {
        return BlingFireUtils.TextToIds(Model, buffer, buffer.Length, tempBuffer, tempBuffer.Length, UnkToken);
    }
}
