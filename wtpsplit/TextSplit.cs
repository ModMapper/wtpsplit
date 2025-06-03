namespace wtpsplit;
using CommunityToolkit.HighPerformance;

using System;
using System.Collections.Generic;

internal struct TextSplit {
    public delegate IEnumerable<ReadOnlyMemory<char>> SplitMethod(ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float threshold);

    public static SplitMethod GetSplit(bool stripWhitespace, bool splitOnInputNewlines) {
        return splitOnInputNewlines
            ? (stripWhitespace
                ? SplitWithNewlineTrim
                : SplitWithNewline)
            : (stripWhitespace
                ? SplitWithTrim
                : Split);

        static IEnumerable<ReadOnlyMemory<char>> Split(ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float threshold) {
            int index = 0;
            do {
                int count = Next(text.Span[index..], probs.Span[index..], threshold);
                if (0 < count) {
                    yield return text.Slice(index, count);
                }
                index += count;
            } while (index < text.Length);
        }

        static IEnumerable<ReadOnlyMemory<char>> SplitWithTrim(ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float threshold) {
            return Trim(Split(text, probs, threshold));
        }

        static IEnumerable<ReadOnlyMemory<char>> SplitWithNewline(ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float threshold) {
            foreach (ReadOnlyMemory<char> subtext in Split(text, probs, threshold)) {
                int index = 0;
                int count = subtext.Span.IndexOf('\n');
                while (0 <= count) {
                    if (0 < count) {
                        yield return subtext.Slice(index, count);
                    }
                    index += count + 1;
                    count = subtext.Span[index..].IndexOf('\n');
                }
                if (index < subtext.Length) {
                    yield return subtext[index..];
                }
            }
        }

        static IEnumerable<ReadOnlyMemory<char>> SplitWithNewlineTrim(ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float threshold) {
            return Trim(SplitWithNewline(text, probs, threshold));
        }
    }

    public static IEnumerable<string> SplitSentence(TextSplit.SplitMethod split, ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float sentenceThreshold) {
        return GetStrings(split(text, probs, sentenceThreshold));
    }

    public static IEnumerable<IEnumerable<string>> SplitParagraph(TextSplit.SplitMethod split, ReadOnlyMemory<char> text, ReadOnlyMemory<float> probs, float paragraphThreshold, float sentenceThreshold) {
        int count = 0;
        while (0 < (count = Next(text.Span, probs.Span, paragraphThreshold))) {
            yield return GetStrings(split(text[..count], probs[..count], sentenceThreshold));
            text = text[count..];
            probs = probs[count..];
        }
    }

    private static IEnumerable<ReadOnlyMemory<char>> Trim(IEnumerable<ReadOnlyMemory<char>> strings) {
        foreach (ReadOnlyMemory<char> str in strings) {
            ReadOnlyMemory<char> trimmed = str.Trim();
            if (!trimmed.IsEmpty) {
                yield return trimmed;
            }
        }
    }

    private static IEnumerable<string> GetStrings(IEnumerable<ReadOnlyMemory<char>> strings) {
        foreach (ReadOnlyMemory<char> str in strings) {
            yield return new string(str.Span);
        }
    }

    private static int Next(ReadOnlySpan<char> text, ReadOnlySpan<float> probs, float threshold) {
        int index = 0;
        while (index < text.Length) {
            if (probs[index++] <= threshold) { continue; }
            for (; index < text.Length && char.IsWhiteSpace(text[index]); index++) ;
            return index;
        }
        return text.Length;
    }
}
