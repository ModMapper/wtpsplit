namespace wtpsplit;

using Microsoft.ML.OnnxRuntime;

/// <summary>SaT (Segment any Text): predicts sentence and paragraph boundaries in text.</summary>
public partial class SaT {
    /// <summary>Initializes a SaT instance with model name, path and tokenizer.</summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="modelPath">Path to the model file.</param>
    /// <param name="tokenizer">Tokenizer instance to use.</param>
    public SaT(string modelName, string modelPath, ITokenizer tokenizer)
        : this(modelName, new InferenceSession(GetModelPath(modelPath)), tokenizer) { }

    /// <summary>Initializes a SaT instance with model name, path, tokenizer, and session options.</summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="modelPath">Path to the model file.</param>
    /// <param name="tokenizer">Tokenizer instance to use.</param>
    /// <param name="options">ONNX session options.</param>
    public SaT(string modelName, string modelPath, ITokenizer tokenizer, SessionOptions options)
        : this(modelName, new InferenceSession(GetModelPath(modelPath), options), tokenizer) { }

    /// <summary>Initializes a SaT instance with model name, raw model bytes and tokenizer.</summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">Raw ONNX model bytes.</param>
    /// <param name="tokenizer">Tokenizer instance to use.</param>
    public SaT(string modelName, byte[] model, ITokenizer tokenizer)
        : this(modelName, new InferenceSession(model), tokenizer) { }
    
    /// <summary>Initializes a SaT instance with model name, raw model bytes, tokenizer, and session options.</summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="model">Raw ONNX model bytes.</param>
    /// <param name="tokenizer">Tokenizer instance to use.</param>
    /// <param name="options">ONNX session options.</param>
    public SaT(string modelName, byte[] model, ITokenizer tokenizer, SessionOptions options)
        : this(modelName, new InferenceSession(model, options), tokenizer) { }

    /// <summary>Initializes a SaT instance with model name, inference session, and tokenizer.</summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="session">ONNX inference session instance.</param>
    /// <param name="tokenizer">Tokenizer instance to use.</param>
    public SaT(string modelName, InferenceSession session, ITokenizer tokenizer) {
        Model = new(session);
        ModelName = modelName;
        Tokenizer = tokenizer;
        ModelThreshold = GetDefaultThreshold(modelName);
    }

    /// <summary>The name of the model used by this SaT instance.</summary>
    public string ModelName { get; }

    private Model Model { get; }

    private ITokenizer Tokenizer { get; }

    private float ModelThreshold { get; }

    private static string GetModelPath(string modelPath) {
        modelPath = Path.GetFullPath(modelPath);

        if (File.Exists(modelPath)) {
            return modelPath;
        }

        if (Directory.Exists(modelPath)) {
            string optimizedPath = Path.Combine(modelPath, "model_optimized.onnx");
            if (File.Exists(optimizedPath)) {
                return optimizedPath;
            }

            string defaultPath = Path.Combine(modelPath, "model.onnx");
            if (File.Exists(defaultPath)) {
                return defaultPath;
            }
        }

        throw new FileNotFoundException($"Model file not found at '{modelPath}'");
    }

    private static float GetDefaultThreshold(string modelName) {
        if(modelName != null) {
            if (modelName.Contains("sm")) {
                return 0.25f;
            } else if (modelName.Contains("no-limited-lookahead")) {
                return 0.01f;
            }
        }
        return 0.025f;
    }
}
