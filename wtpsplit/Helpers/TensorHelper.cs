namespace wtpsplit.Utils;
using CommunityToolkit.HighPerformance;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using System;
using System.Runtime.CompilerServices;

internal static class TensorHelper {
    public static Float16[,] HalfToFloat16(Half[,] array) {
        return Unsafe.As<Half[,], Float16[,]>(ref array);
    }

    public static Float16[,,] HalfToFloat16(Half[,,] array) {
        return Unsafe.As<Half[,,], Float16[,,]>(ref array);
    }

    public static DenseTensor<T> View<T>(T[,] array) {
        return new DenseTensor<T>(array.AsMemory(), [array.GetLength(0), array.GetLength(1)]);
    }

    public static DenseTensor<T> View<T>(T[,,] array) {
        return new DenseTensor<T>(array.AsMemory(), [array.GetLength(0), array.GetLength(1), array.GetLength(2)]);
    }

    public static DenseTensor<Float16> ViewHalf(Half[,] array) {
        return View(HalfToFloat16(array));
    }

    public static DenseTensor<Float16> ViewHalf(Half[,,] array) {
        return View(HalfToFloat16(array));
    }
}
