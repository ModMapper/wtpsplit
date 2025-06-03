#pragma warning disable IDE0079
#pragma warning disable SYSLIB1054
// <copyright file="BlingFireUtils.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
//     Licensed under the MIT License.
// </copyright>

namespace wtpsplit.BlingFire {
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Text;

    /// <summary>
    /// This is C# interface for blingfiretokdll.dll, it uses Span<T> hence more efficient however 
    /// is not supported by older .Net frameworks
    /// 
    /// For API description please see blingfiretokdll.cpp comments.
    ///
    /// </summary>
    internal static class BlingFireUtils {
        private const string BlingFireTokDllName = "blingfiretokdll";

        [DllImport(BlingFireTokDllName)]
        public static extern int GetBlingFireTokVersion();

        [DllImport(BlingFireTokDllName)]
        public static extern ulong SetModel(byte[] modelBytes, int modelByteCount);

        [DllImport(BlingFireTokDllName, BestFitMapping = false)]
        public static extern ulong LoadModel([MarshalAs(UnmanagedType.LPStr)] string modelName);

        [DllImport(BlingFireTokDllName)]
        public static extern int FreeModel(ulong model);

        public static IEnumerable<string> GetSentences(string paragraph) {
            // use Bling Fire TOK for sentence breaking
            byte[] paraBytes = Encoding.UTF8.GetBytes(paragraph);
            int maxLength = 2 * paraBytes.Length + 1;
            byte[] outputBytes = new byte[maxLength];

            // native call returns '\n' delimited sentences, and adds 0 byte at the end
            int actualLength = TextToSentences(paraBytes, paraBytes.Length, outputBytes, maxLength);
            if (0 < actualLength - 1 && actualLength <= maxLength) {
                string sentencesStr = Encoding.UTF8.GetString(outputBytes, 0, actualLength);
                var sentences = sentencesStr.Split(g_justNewLineChar, StringSplitOptions.RemoveEmptyEntries);
                foreach (var s in sentences) {
                    yield return s;
                }
            }
        }

        public static IEnumerable<Tuple<string, int, int>> GetSentencesWithOffsets(string paragraph) {
            // use Bling Fire TOK for sentence breaking
            return GetSentencesWithOffsets(Encoding.UTF8.GetBytes(paragraph));
        }

        public static IEnumerable<Tuple<string, int, int>> GetSentencesWithOffsets(byte[] paraBytes) {
            // use Bling Fire TOK for sentence breaking
            int maxLength = 2 * paraBytes.Length + 1;
            byte[] outputBytes = new byte[maxLength];
            int[] startOffsets = new int[maxLength];
            int[] endOffsets = new int[maxLength];

            // native call returns '\n' delimited sentences, and adds 0 byte at the end
            int actualLength = TextToSentencesWithOffsets(paraBytes, paraBytes.Length, outputBytes, startOffsets, endOffsets, maxLength);
            if (0 < actualLength - 1 && actualLength <= maxLength) {
                string sentencesStr = Encoding.UTF8.GetString(outputBytes, 0, actualLength);
                var sentences = sentencesStr.Split(g_justNewLineChar, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < sentences.Length; ++i) {
                    yield return new Tuple<string, int, int>(sentences[i], startOffsets[i], endOffsets[i]);
                }
            }
        }

        public static IEnumerable<string> GetWords(string sentence) {
            // use Bling Fire TOK for sentence breaking
            byte[] paraBytes = Encoding.UTF8.GetBytes(sentence);
            int maxLength = 2 * paraBytes.Length + 1;
            byte[] outputBytes = new byte[maxLength];

            // native call returns '\n' delimited sentences, and adds 0 byte at the end
            int actualLength = TextToWords(paraBytes, paraBytes.Length, outputBytes, maxLength);
            if (0 < actualLength - 1 && actualLength <= maxLength) {
                string wordsStr = Encoding.UTF8.GetString(outputBytes, 0, actualLength);
                var words = wordsStr.Split(g_justSpaceChar, StringSplitOptions.RemoveEmptyEntries);
                foreach (var w in words) {
                    yield return w;
                }
            }
        }

        public static IEnumerable<Tuple<string, int, int>> GetWordsWithOffsets(string sentence) {
            // use Bling Fire TOK for sentence breaking
            byte[] paraBytes = Encoding.UTF8.GetBytes(sentence);
            int maxLength = 2 * paraBytes.Length + 1;
            byte[] outputBytes = new byte[maxLength];
            int[] startOffsets = new int[maxLength];
            int[] endOffsets = new int[maxLength];

            // native call returns '\n' delimited sentences, and adds 0 byte at the end
            int actualLength = TextToWordsWithOffsets(paraBytes, paraBytes.Length, outputBytes, startOffsets, endOffsets, maxLength);
            if (0 < actualLength - 1 && actualLength <= maxLength) {
                string wordsStr = Encoding.UTF8.GetString(outputBytes, 0, actualLength);
                var words = wordsStr.Split(g_justSpaceChar, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < words.Length; ++i) {
                    yield return new Tuple<string, int, int>(words[i], startOffsets[i], endOffsets[i]);
                }
            }
        }

        public static string IdsToText(ulong model, int[] ids, bool skipSpecialTokens = true) {
            if (null == ids || 0 == ids.Length) {
                return string.Empty;
            }

            // guess maximum needed buffer size
            int MaxOutputSize = Math.Max(4096, ids.Length * 32);
            byte[] outputBytes = new byte[MaxOutputSize];
            int actualLength = IdsToText(model, ids, ids.Length, outputBytes, outputBytes.Length, skipSpecialTokens);

            // if the buffer is too small call it again with a bigger buffer
            if (0 < actualLength && actualLength > outputBytes.Length) {
                outputBytes = new byte[actualLength];
                actualLength = IdsToText(model, ids, ids.Length, outputBytes, outputBytes.Length, skipSpecialTokens);
            }

            // see if the results are ready
            if (0 < actualLength && actualLength <= outputBytes.Length) {
                return Encoding.UTF8.GetString(outputBytes, 0, actualLength);
            }

            return string.Empty;
        }


        //
        // expose Bling Fire interfaces
        //
        [DllImport(BlingFireTokDllName)]
        static extern int TextToSentences(in byte inUtf8Str, int inUtf8StrLen, ref byte outBuff, int maxBuffSize);

        public static int TextToSentences(Span<byte> inUtf8Str, int inUtf8StrLen, Span<byte> outBuff, int maxBuffSize) {
            return TextToSentences(
                in MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToWords(in byte inUtf8Str, int inUtf8StrLen, ref byte outBuff, int maxBuffSize);

        public static int TextToWords(Span<byte> inUtf8Str, int inUtf8StrLen, Span<byte> outBuff, int maxBuffSize) {
            return TextToWords(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToSentencesWithModel(in byte inUtf8Str, int inUtf8StrLen, ref byte outBuff, int maxBuffSize, ulong model);

        public static int TextToSentencesWithModel(Span<byte> inUtf8Str, int inUtf8StrLen, Span<byte> outBuff, int maxBuffSize, ulong model) {
            return TextToSentencesWithModel(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize,
                model);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToWordsWithModel(in byte inUtf8Str, int inUtf8StrLen, ref byte outBuff, int maxBuffSize, ulong model);

        public static int TextToWordsWithModel(Span<byte> inUtf8Str, int inUtf8StrLen, Span<byte> outBuff, int maxBuffSize, ulong model) {
            return TextToWordsWithModel(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize,
                model);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToSentencesWithOffsets(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            ref int startOffsets,
            ref int endOffsets,
            int maxBuffSize);

        public static int TextToSentencesWithOffsets(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff,
            Span<int> startOffsets,
            Span<int> endOffsets,
            int maxBuffSize) {
            return TextToSentencesWithOffsets(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                ref MemoryMarshal.GetReference(startOffsets),
                ref MemoryMarshal.GetReference(endOffsets),
                maxBuffSize);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToWordsWithOffsets(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            ref int startOffsets,
            ref int endOffsets,
            int maxBuffSize);

        public static int TextToWordsWithOffsets(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff,
            Span<int> startOffsets,
            Span<int> endOffsets,
            int maxBuffSize) {
            return TextToWordsWithOffsets(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                ref MemoryMarshal.GetReference(startOffsets),
                ref MemoryMarshal.GetReference(endOffsets),
                maxBuffSize);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToSentencesWithOffsetsWithModel(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            ref int startOffsets,
            ref int endOffsets,
            int maxBuffSize,
            ulong model);

        public static int TextToSentencesWithOffsetsWithModel(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff,
            Span<int> startOffsets,
            Span<int> endOffsets,
            int maxBuffSize,
            ulong model) {
            return TextToSentencesWithOffsetsWithModel(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                ref MemoryMarshal.GetReference(startOffsets),
                ref MemoryMarshal.GetReference(endOffsets),
                maxBuffSize,
                model);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToWordsWithOffsetsWithModel(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            ref int startOffsets,
            ref int endOffsets,
            int maxBuffSize,
            ulong model);

        public static int TextToWordsWithOffsetsWithModel(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff,
            Span<int> startOffsets,
            Span<int> endOffsets,
            int maxBuffSize,
            ulong model) {
            return TextToWordsWithOffsetsWithModel(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                ref MemoryMarshal.GetReference(startOffsets),
                ref MemoryMarshal.GetReference(endOffsets),
                maxBuffSize,
                model);
        }


        [DllImport(BlingFireTokDllName)]
        static extern int WordHyphenationWithModel(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            int maxBuffSize,
            ulong model,
            int utf32HyCode);

        public static int WordHyphenationWithModel(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff, int maxBuffSize,
            ulong model,
            int utf32HyCode = 0x2D) {
            return WordHyphenationWithModel(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize,
                model,
                utf32HyCode);
        }


        [DllImport(BlingFireTokDllName)]
        static extern int TextToIds(
            ulong model,
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref int tokenIds,
            int maxBuffSize,
            int unkId);

        public static int TextToIds(
            ulong model,
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<int> tokenIds,
            int maxBuffSize,
            int unkId) {
            return TextToIds(
                model,
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(tokenIds),
                maxBuffSize,
                unkId);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int TextToIdsWithOffsets(
           ulong model,
           in byte inUtf8Str,
           int inUtf8StrLen,
           ref int tokenIds,
           ref int startOffsets,
           ref int endOffsets,
           int maxBuffSize,
           int unkId);

        public static int TextToIdsWithOffsets(
            ulong model,
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<int> tokenIds,
            Span<int> startOffsets,
            Span<int> endOffsets,
            int maxBuffSize,
            int unkId) {
            return TextToIdsWithOffsets(
                model,
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(tokenIds),
                ref MemoryMarshal.GetReference(startOffsets),
                ref MemoryMarshal.GetReference(endOffsets),
                maxBuffSize,
                unkId);
        }

        [DllImport(BlingFireTokDllName)]
        static extern int NormalizeSpaces(
            in byte inUtf8Str,
            int inUtf8StrLen,
            ref byte outBuff,
            int maxBuffSize,
            int utf32SpaceCode);

        public static int NormalizeSpaces(
            Span<byte> inUtf8Str,
            int inUtf8StrLen,
            Span<byte> outBuff,
            int maxBuffSize,
            int utf32SpaceCode) {
            return NormalizeSpaces(
                MemoryMarshal.GetReference(inUtf8Str),
                inUtf8StrLen,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize,
                utf32SpaceCode);
        }

        [DllImport(BlingFireTokDllName)]
        public static extern int SetNoDummyPrefix(ulong model, bool fNoDummyPrefix);

        [DllImport(BlingFireTokDllName)]
        static extern int IdsToText(
            ulong model,
            in int ids,
            int idsCount,
            ref byte outBuff,
            int maxBuffSize,
            bool skipSpecialTokens);

        public static int IdsToText(
            ulong model,
            Span<int> ids,
            int idsCount,
            Span<byte> outBuff,
            int maxBuffSize,
            bool skipSpecialTokens) {
            return IdsToText(
                model,
                MemoryMarshal.GetReference(ids),
                idsCount,
                ref MemoryMarshal.GetReference(outBuff),
                maxBuffSize,
                skipSpecialTokens);
        }


        private static readonly char[] g_justNewLineChar = ['\n'];
        private static readonly char[] g_justSpaceChar = [' '];
    }
}