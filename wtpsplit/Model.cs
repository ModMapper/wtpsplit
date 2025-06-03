namespace wtpsplit;

using Microsoft.ML.OnnxRuntime;

using System;

using wtpsplit.Utils;

internal class Model {
    public Model(InferenceSession session) {
        Session = session;
        NumLabels = Session.OutputMetadata["logits"].Dimensions[2];
        DownsamplingRate = Constants.DEFAULT_DOWNSAMPLING_RATE;
    }

    public InferenceSession Session { get; }

    public int NumLabels { get; }

    public int DownsamplingRate { get; }

    public void Inference(long[,] inputIds, Half[,] attentionMask, Half[,,] logits) {
        IReadOnlyCollection<NamedOnnxValue> input = [
            NamedOnnxValue.CreateFromTensor("input_ids", TensorHelper.View(inputIds)),
            NamedOnnxValue.CreateFromTensor("attention_mask", TensorHelper.ViewHalf(attentionMask)),
        ];
        IReadOnlyCollection<NamedOnnxValue> output = [
            NamedOnnxValue.CreateFromTensor("logits", TensorHelper.ViewHalf(logits))
        ];
        Session.Run(input, output);
    }
}
