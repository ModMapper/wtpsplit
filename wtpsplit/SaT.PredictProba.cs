namespace wtpsplit;

using CommunityToolkit.HighPerformance;

using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

using wtpsplit.Utils;

public partial class SaT {
    /// <summary>
    /// Predicts the probability of end-of-sentence for each character in given texts.
    /// </summary>
    /// <param name="texts">List of input texts.</param>
    /// <param name="stride">Stride size for token windowing.</param>
    /// <param name="blockSize">Maximum token block size.</param>
    /// <param name="batchSize">Batch size for model inference.</param>
    /// <param name="padLastBatch">Whether to pad the last batch.</param>
    /// <param name="weighting">Method used to combine logits.</param>
    /// <param name="removeWhitespaceBeforeInference">Whether to remove whitespaces before inference.</param>
    /// <param name="outerBatchSize">Batch size for splitting input texts.</param>
    /// <returns>Sequence of float arrays representing newline probabilities per character.</returns>
    public IEnumerable<float[]> PredictProba(
        List<string> texts,
        int stride = 256,
        int blockSize = 512,
        int batchSize = 32,
        bool padLastBatch = false,
        WeightingType weighting = WeightingType.Uniform,
        bool removeWhitespaceBeforeInference = false,
        int outerBatchSize = 1000) {

        int nOuterBatches = MathHelper.CeilDivide(texts.Count, outerBatchSize);

        for (int outerBatchIdx = 0; outerBatchIdx < nOuterBatches; outerBatchIdx++) {
            int start = outerBatchIdx * outerBatchSize;
            int end = Math.Min(start + outerBatchSize, texts.Count);

            List<string> outerBatchTexts = texts[start..end];
            List<string> inputTexts;
            List<List<int>> spacePositions = [];

            // Remove whitespaces and record their positions for later restoration
            if (removeWhitespaceBeforeInference) {
                StringBuilder inputText = new();
                inputTexts = [];

                foreach(string text in outerBatchTexts) {
                    List<int> textSpacePositions = [];
                    inputText.Clear();
                    for(int i = 0; i < text.Length; i++) {
                        char ch = text[i];
                        if (ch == ' ') {
                            textSpacePositions.Add(i);
                        } else {
                            inputText.Append(ch);
                        }
                    }
                    spacePositions.Add(textSpacePositions);
                    inputTexts.Add(inputText.ToString());
                }
            } else {
                inputTexts = outerBatchTexts;
            }

            // Track indices of empty or whitespace-only strings
            List<int> emptyStringIndices = [];
            for (int i = 0; i < inputTexts.Count; i++) {
                if (string.IsNullOrWhiteSpace(inputTexts[i])) {
                    emptyStringIndices.Add(i);
                }
            }

            // Filter out empty strings from inputTexts
            inputTexts.RemoveAll(string.IsNullOrWhiteSpace);

            // Extract logits for the non-empty strings
            List<float[,]> outerBatchLogits = new(inputTexts.Count);
            if (0 < inputTexts.Count) {
                ProbaExtract.TokenLogits[] tokenLogits = ProbaExtract.Extract(
                    inputTexts,
                    Model,
                    stride,
                    blockSize,
                    batchSize,
                    padLastBatch,
                    weighting,
                    Tokenizer);
                for (int i = 0; i < inputTexts.Count; i++) {
                    // Store character-level logits for each input text
                    outerBatchLogits.Add(TokenToCharProbs(
                        inputTexts[i],
                        tokenLogits[i].Tokens,
                        tokenLogits[i].Logits,
                        Tokenizer));
                }
            }

            // Restore empty strings with negative infinity logits
            foreach (var idx in emptyStringIndices) {
                outerBatchLogits.Insert(idx, new float[,] { { float.NegativeInfinity } });
            }

            for(int i = 0; i < outerBatchTexts.Count; i++) {
                var text = outerBatchTexts[i];
                var logits = outerBatchLogits[i];

                var sentenceProbs = NewlineProbabilityFn(logits);

                if (removeWhitespaceBeforeInference) {
                    // Restore whitespace positions in the sentence probabilities
                    List<int> spacePosition = spacePositions[i];
                    if (0 < spacePosition.Count) {
                        float[] fullSentenceProbs = new float[text.Length + spacePosition.Count];
                        int srcIdx = 0, dstIdx = 0;
                        foreach(int position in spacePosition) {
                            int length = position - dstIdx;
                            Array.Copy(sentenceProbs, srcIdx, fullSentenceProbs, dstIdx, length);
                            srcIdx += length;
                            dstIdx += length + 1;
                        }
                        Array.Copy(sentenceProbs, srcIdx, fullSentenceProbs, dstIdx, sentenceProbs.Length - srcIdx);
                        sentenceProbs = fullSentenceProbs;
                    }
                }

                yield return sentenceProbs;
            }
        }

        static float[,] TokenToCharProbs(string text, ITokenizer.Token[] tokens, float[,] tokenLogits, ITokenizer tokenizer) {
            float[,] charProbs = new float[text.Length, tokenLogits.GetLength(1)];
            charProbs.AsSpan().Fill(float.NegativeInfinity);
            // Copy token logits to character logits
            foreach (var (index, offset) in GetTokenSpans(tokenizer, tokens)) {
                tokenLogits.GetRowSpan(index).CopyTo(charProbs.GetRowSpan(Math.Max(offset.End - 1, 0)));
            }
            return charProbs;
        }

        static IEnumerable<(int Index, ITokenizer.Offset Offset)> GetTokenSpans(ITokenizer tokenizer, ITokenizer.Token[] tokens) {
            int bosToken = tokenizer.BosToken;
            int eosToken = tokenizer.EosToken;
            int padToken = tokenizer.PadToken;
            Span<int> specialTokens = [tokenizer.BosToken, tokenizer.EosToken, tokenizer.PadToken];
            for (int idx = 0; idx < tokens.Length; idx++) {
                int tokenId = tokens[idx].TokenId;
                if (tokenId != bosToken && tokenId != eosToken && tokenId != padToken) {
                    yield return (idx, tokens[idx].Offset);
                }
            }
        }

        static float[] NewlineProbabilityFn(float[,] logits) {
            float[] probs = logits.GetColumn(Constants.NEWLINE_INDEX).ToArray();
            TensorPrimitives.Sigmoid(probs, probs);
            return probs;
        }
    }
}
