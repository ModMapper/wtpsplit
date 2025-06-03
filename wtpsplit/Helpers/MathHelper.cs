namespace wtpsplit.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;

internal static class MathHelper {
    public static int CeilDivide(int x, int y) {
        return (x + y - 1) / y;
    }

    public static float[] Hat(int points) {
        float x = 1 - 1.0f / points;
        float[] weights = Linspace(-x, x, points);
        TensorPrimitives.Abs(weights, weights);
        TensorPrimitives.Subtract(1f, weights, weights);
        return weights;
    }

    public static float[] Linspace(float start, float end, int points) {
        float[] result = new float[points];
        float step = (end - start) / (points - 1);
        for (int i = 0; i < points; i++) {
            result[i] = start + step * i;
        }
        return result;
    }
}
