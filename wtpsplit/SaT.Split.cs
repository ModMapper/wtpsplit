namespace wtpsplit;

using System;
using System.Collections.Generic;
using System.Linq;

public partial class SaT {
    /// <summary>
    /// Splits the input text into sentences using predicted newline probabilities.
    /// </summary>
    /// <param name="text">Input text to split.</param>
    /// <param name="threshold">Probability threshold to decide sentence boundaries.</param>
    /// <param name="stride">Stride size for token windowing.</param>
    /// <param name="blockSize">Maximum token block size.</param>
    /// <param name="batchSize">Batch size for model inference.</param>
    /// <param name="padLastBatch">Whether to pad the last batch.</param>
    /// <param name="weighting">Method used to combine logits.</param>
    /// <param name="removeWhitespaceBeforeInference">Whether to remove whitespaces before inference.</param>
    /// <param name="outerBatchSize">Batch size for splitting input texts.</param>
    /// <param name="stripWhitespace">Whether to trim whitespace from split sentences.</param>
    /// <param name="splitOnInputNewlines">Whether to treat existing newlines as split points.</param>
    /// <returns>Sequence of split sentence strings.</returns>
    public IEnumerable<string> Split(
        string text,
        float? threshold = null,
        int stride = 64,
        int blockSize = 512,
        int batchSize = 32,
        bool padLastBatch = false,
        WeightingType weighting = WeightingType.Uniform,
        bool removeWhitespaceBeforeInference = false,
        int outerBatchSize = 1000,
        bool stripWhitespace = false,
        bool splitOnInputNewlines = true) {

        float sentenceThreshold = threshold ?? ModelThreshold;

        float[] probs = PredictProba(
            [text],
            stride,
            blockSize,
            batchSize,
            padLastBatch,
            weighting,
            removeWhitespaceBeforeInference,
            outerBatchSize).First();

        TextSplit.SplitMethod split = TextSplit.GetSplit(stripWhitespace, splitOnInputNewlines);
        return TextSplit.SplitSentence(split, text.AsMemory(), probs, sentenceThreshold);
    }

    /// <summary>
    /// Splits multiple input texts into sentences using predicted newline probabilities.
    /// </summary>
    /// <param name="texts">List of input texts to split.</param>
    /// <param name="threshold">Probability threshold to decide sentence boundaries.</param>
    /// <param name="stride">Stride size for token windowing.</param>
    /// <param name="blockSize">Maximum token block size.</param>
    /// <param name="batchSize">Batch size for model inference.</param>
    /// <param name="padLastBatch">Whether to pad the last batch.</param>
    /// <param name="weighting">Method used to combine logits.</param>
    /// <param name="removeWhitespaceBeforeInference">Whether to remove whitespaces before inference.</param>
    /// <param name="outerBatchSize">Batch size for splitting input texts.</param>
    /// <param name="stripWhitespace">Whether to trim whitespace from split sentences.</param>
    /// <param name="splitOnInputNewlines">Whether to treat existing newlines as split points.</param>
    /// <returns>Sequence of sentence lists for each input text.</returns>
    public IEnumerable<IEnumerable<string>> Split(
        List<string> texts,
        float? threshold = null,
        int stride = 64,
        int blockSize = 512,
        int batchSize = 32,
        bool padLastBatch = false,
        WeightingType weighting = WeightingType.Uniform,
        bool removeWhitespaceBeforeInference = false,
        int outerBatchSize = 1000,
        bool stripWhitespace = false,
        bool splitOnInputNewlines = true) {

        float sentenceThreshold = threshold ?? ModelThreshold;

        IEnumerable<float[]> predict = PredictProba(
            texts,
            stride,
            blockSize,
            batchSize,
            padLastBatch,
            weighting,
            removeWhitespaceBeforeInference,
            outerBatchSize);

        TextSplit.SplitMethod split = TextSplit.GetSplit(stripWhitespace, splitOnInputNewlines);
        foreach (var (text, probs) in texts.Zip(predict)) {
            yield return TextSplit.SplitSentence(split, text.AsMemory(), probs, sentenceThreshold);
        }
    }
    /// <summary>
    /// Splits the input text into paragraphs, each consisting of multiple sentences,
    /// using predicted newline and paragraph probabilities.
    /// </summary>
    /// <param name="text">Input text to split into paragraphs.</param>
    /// <param name="threshold">Probability threshold to decide sentence boundaries.</param>
    /// <param name="paragraphThreshold">Threshold to decide paragraph boundaries between sentences.</param>
    /// <param name="stride">Stride size for token windowing.</param>
    /// <param name="blockSize">Maximum token block size.</param>
    /// <param name="batchSize">Batch size for model inference.</param>
    /// <param name="padLastBatch">Whether to pad the last batch.</param>
    /// <param name="weighting">Method used to combine logits.</param>
    /// <param name="removeWhitespaceBeforeInference">Whether to remove whitespaces before inference.</param>
    /// <param name="outerBatchSize">Batch size for splitting input texts.</param>
    /// <param name="stripWhitespace">Whether to trim whitespace from split sentences.</param>
    /// <param name="splitOnInputNewlines">Whether to treat existing newlines as split points.</param>
    /// <returns>Sequence of paragraphs, each represented as a list of sentence strings.</returns>
    public IEnumerable<IEnumerable<string>> SplitParagraph(
        string text,
        float? threshold = null,
        float paragraphThreshold = 0.5f,
        int stride = 64,
        int blockSize = 512,
        int batchSize = 32,
        bool padLastBatch = false,
        WeightingType weighting = WeightingType.Uniform,
        bool removeWhitespaceBeforeInference = false,
        int outerBatchSize = 1000,
        bool stripWhitespace = false,
        bool splitOnInputNewlines = true) {

        float sentenceThreshold = threshold ?? ModelThreshold;

        float[] probs = PredictProba(
            [text],
            stride,
            blockSize,
            batchSize,
            padLastBatch,
            weighting,
            removeWhitespaceBeforeInference,
            outerBatchSize).First();

        TextSplit.SplitMethod split = TextSplit.GetSplit(stripWhitespace, splitOnInputNewlines);
        return TextSplit.SplitParagraph(split, text.AsMemory(), probs, paragraphThreshold, sentenceThreshold);
    }
    /// <summary>
    /// Splits multiple input texts into paragraphs, each consisting of multiple sentences,
    /// using predicted newline and paragraph probabilities.
    /// </summary>
    /// <param name="texts">List of input texts to split into paragraphs.</param>
    /// <param name="threshold">Probability threshold to decide sentence boundaries.</param>
    /// <param name="paragraphThreshold">Threshold to decide paragraph boundaries between sentences.</param>
    /// <param name="stride">Stride size for token windowing.</param>
    /// <param name="blockSize">Maximum token block size.</param>
    /// <param name="batchSize">Batch size for model inference.</param>
    /// <param name="padLastBatch">Whether to pad the last batch.</param>
    /// <param name="weighting">Method used to combine logits.</param>
    /// <param name="removeWhitespaceBeforeInference">Whether to remove whitespaces before inference.</param>
    /// <param name="outerBatchSize">Batch size for splitting input texts.</param>
    /// <param name="stripWhitespace">Whether to trim whitespace from split sentences.</param>
    /// <param name="splitOnInputNewlines">Whether to treat existing newlines as split points.</param>
    /// <returns>List of paragraphs (as sentence lists) per input text.</returns>
    public IEnumerable<IEnumerable<IEnumerable<string>>> SplitParagraph(
        List<string> texts,
        float? threshold = null,
        float paragraphThreshold = 0.5f,
        int stride = 64,
        int blockSize = 512,
        int batchSize = 32,
        bool padLastBatch = false,
        WeightingType weighting = WeightingType.Uniform,
        bool removeWhitespaceBeforeInference = false,
        int outerBatchSize = 1000,
        bool stripWhitespace = false,
        bool splitOnInputNewlines = true) {

        float sentenceThreshold = threshold ?? ModelThreshold;

        IEnumerable<float[]> predict = PredictProba(
            texts,
            stride,
            blockSize,
            batchSize,
            padLastBatch,
            weighting,
            removeWhitespaceBeforeInference,
            outerBatchSize);

        TextSplit.SplitMethod split = TextSplit.GetSplit(stripWhitespace, splitOnInputNewlines);
        foreach (var (text, probs) in texts.Zip(predict)) {
            yield return TextSplit.SplitParagraph(split, text.AsMemory(), probs, paragraphThreshold, sentenceThreshold);
        }
    }
}
