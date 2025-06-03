namespace wtpsplit;
using CommunityToolkit.HighPerformance;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using wtpsplit.Utils;

using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

internal static class ProbaExtract {
    public static TokenLogits[] Extract(
            List<string> inputTexts,
            Model model,
            int stride,
            int maxBlockSize,
            int batchSize,
            bool padLastBatch,
            WeightingType weighting,
            ITokenizer tokenizer) {

        var inputTokens = inputTexts.ConvertAll(tokenizer.Encode);
        int clsTokenId = tokenizer.BosToken;
        int sepTokenId = tokenizer.EosToken;
        int padTokenId = tokenizer.PadToken;

        int blockSize = Math.Min(Constants.MAX_BLOCK_SIZE, Math.Min(maxBlockSize, inputTokens.Max((v) => v.Count)));
        int downsamplingRate = model.DownsamplingRate;
        blockSize = MathHelper.CeilDivide(blockSize, downsamplingRate) * downsamplingRate;

        int numChunks = inputTokens.Sum((t) => MathHelper.CeilDivide(Math.Max(t.Count - blockSize, 0), stride) + 1);

        int         blockWidth      = blockSize + 2;
        long[,]     inputIds        = new long[numChunks, blockWidth];
        Half[,]     attentionMask   = new Half[numChunks, blockWidth];
        CharIndex[] locs            = new CharIndex[numChunks];

        int currentChunk = 0;
        for(int i = 0; i < inputTokens.Count; i++) {
            int[] batchTokens = inputTokens[i].Select((token) => token.TokenId).ToArray();
            int length = inputTokens[i].Count;

            for (int j = 0; j < length; j += stride) {
                int start = j;
                int end = j + blockSize;
                bool done = false;

                if (end >= length) {
                    end = length;
                    start = Math.Max(end - blockSize, 0);
                    done = true;
                }

                // Slice the tokens for the current chunk
                var inputIdsRow = inputIds.GetRowSpan(currentChunk);
                var tokens = batchTokens.AsSpan(start..end);
                inputIdsRow[0] = clsTokenId;
                TensorPrimitives.ConvertSaturating<int, long>(tokens, inputIdsRow[1..]);
                inputIdsRow[tokens.Length - 1] = sepTokenId;
                attentionMask.GetRowSpan(currentChunk)[..(tokens.Length + 2)].Fill(Half.One);

                // Add current chunk location
                locs[currentChunk] = new(i, start, end);
                currentChunk++;

                if (done) {
                    break;
                }
            }
        }

        Debug.Assert(currentChunk == numChunks);
        int         numLabels = model.NumLabels;
        float[][]   allCounts = new float[inputTokens.Count][];
        float[][,]  allLogits = new float[inputTokens.Count][,];
        for (int i = 0; i < inputTokens.Count; i++) {
            allCounts[i] = new float[inputTokens[i].Count];
            allLogits[i] = new float[inputTokens[i].Count, numLabels];
        }

        // Initlize Weights
        float[] weights = GetWeights(weighting, blockSize);

        // Run inference in batches
        int nBatches = MathHelper.CeilDivide(numChunks, batchSize);
        for (int batchIdx = 0; batchIdx < nBatches; batchIdx++) {
            int start   = batchIdx * batchSize;
            int end     = Math.Min(numChunks, start + batchSize);
            int size    = end - start;
            long[,]     batchInputIds;
            Half[,]     batchAttentionMask;
            Half[,,]    outputLogits;

            // Pad last batch if necessary
            if (padLastBatch && size < batchSize) {
                batchInputIds = new long[batchSize, blockWidth];
                inputIds.AsSpan2D(start, 0, size, blockWidth).CopyTo(batchInputIds);
                batchInputIds.AsSpan2D()[size.., ..].Fill(padTokenId);

                batchAttentionMask = new Half[batchSize, blockWidth];
                attentionMask.AsSpan2D(start, 0, size, blockWidth).CopyTo(batchAttentionMask);

                outputLogits = new Half[batchSize, blockWidth, numLabels];
            } else {
                batchInputIds = new long[size, blockWidth];
                inputIds.AsSpan2D(start, 0, size, blockWidth).CopyTo(batchInputIds);

                batchAttentionMask = new Half[size, blockWidth];
                attentionMask.AsSpan2D(start, 0, size, blockWidth).CopyTo(batchAttentionMask);

                outputLogits = new Half[size, blockWidth, numLabels];
            }

            // Run inference
            model.Inference(batchInputIds, batchAttentionMask, outputLogits);

            // Process output logits
            for (int i = 0; i < size; i++) {
                // Cut off CLS and SEP tokens
                Span2D<Half> logits = outputLogits.AsSpan2D(i)[1..^1, ..];

                // Get indices and character positions
                var (originalIdx, startCharIdx, endCharIdx) = locs[start + i];

                // Update logits for each sequence in the batch
                UpdateLogits(allLogits, allCounts, weights, logits, originalIdx, startCharIdx, endCharIdx);
            }
        }

        // Average logits
        TokenLogits[] updatedLogits = new TokenLogits[allLogits.Length];
        for (int i = 0; i < allLogits.Length; i++) {
            var logits = allLogits[i];
            var counts = allCounts[i];

            // Average the logits by dividing by counts
            float[,] averaged = new float[counts.Length, numLabels];
            for (int r = 0; r < counts.Length; r++) {
                TensorPrimitives.Divide(logits.GetRowSpan(r), counts[r], averaged.GetRowSpan(r));
            }

            updatedLogits[i] = new(averaged, [.. inputTokens[i]]);
        }
        return updatedLogits;
    }

    private static float[] GetWeights(WeightingType weighting, int blockSize) {
        switch (weighting) {
        case WeightingType.Uniform:
            float[] weights = new float[blockSize];
            Array.Fill(weights, 1f);
            return weights;
        case WeightingType.Hat:
            return MathHelper.Hat(blockSize);
        default: throw new InvalidOperationException();
        }
    }

    private static void UpdateLogits(float[][,] allLogits, float[][] allCounts, float[] weights, Span2D<Half> logits, int originalIdx, int startCharIdx, int endCharIdx) {
        Span<float> logitsRow = stackalloc float[logits.Width];
        int n = endCharIdx - startCharIdx;
        for (int j = 0; j < n; j++) {
            Span<float> allLogitsRow = allLogits[originalIdx].GetRowSpan(startCharIdx + j);
            TensorPrimitives.ConvertToSingle(logits.GetRowSpan(j), logitsRow);
            TensorPrimitives.FusedMultiplyAdd(logitsRow, weights[j], allLogitsRow, allLogitsRow);
        }
        Span<float> allCountsRow = allCounts[originalIdx].AsSpan(startCharIdx, n);
        TensorPrimitives.Add(allCountsRow, weights.AsSpan(0, n), allCountsRow);
    }

    private record struct CharIndex(int Idx, int Start, int End);

    public record struct TokenLogits(float[,] Logits, ITokenizer.Token[] Tokens);
}
 