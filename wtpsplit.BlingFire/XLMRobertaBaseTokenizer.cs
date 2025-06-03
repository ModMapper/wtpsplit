namespace wtpsplit.BlingFire;

using wtpsplit.BlingFire.Properties;

/// <summary>Tokenizer implementation for XLM-RoBERTa Base using BlingFire backend.</summary>
public class XLMRobertaBaseTokenizer : BlingFireTokenizer {

    /// <summary>Initializes the tokenizer with XLM-RoBERTa Base model and vocabulary size.</summary>
    public XLMRobertaBaseTokenizer() : base(Resources.XLMRobertaBase, 0x10000) { }

    /// <summary>Token ID for unknown tokens.</summary>
    public override int UnkToken => 3;

    /// <summary>Token ID for beginning-of-sequence (BOS).</summary>
    public override int BosToken => 0;

    /// <summary>Token ID for end-of-sequence (EOS).</summary>
    public override int EosToken => 2;

    /// <summary>Token ID used for padding.</summary>
    public override int PadToken => 1;
}
