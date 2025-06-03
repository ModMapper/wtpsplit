using HuggingfaceHub;

using Microsoft.ML.OnnxRuntime;

using System.Diagnostics;

using wtpsplit;
using wtpsplit.BlingFire;

// Model name for the SaT model
const string modelName = "segment-any-text/sat-12l";

// Set console output encoding to UTF-8 to handle special characters
Console.OutputEncoding = System.Text.Encoding.UTF8;

// Download the model file from Hugging Face Hub
//string modelPath = await HFDownloader.DownloadSnapshotAsync(modelName);
string modelPath = await HFDownloader.DownloadFileAsync(modelName, "model_optimized.onnx");

// Set up the ONNX Runtime session options
SessionOptions options = new SessionOptions();
options.AppendExecutionProvider_CUDA();

// Create XLM-RoBERTa tokenizer
ITokenizer tokenizer = new XLMRobertaBaseTokenizer();

// Create SaT instance with the model and tokenizer
SaT sat = new SaT(modelName, modelPath, tokenizer, options);

// Read the input text from a file
string text = File.ReadAllText("input.txt");

// Split the text into sentences using the SaT model
int index = 1;
Stopwatch stopwatch = Stopwatch.StartNew();
foreach (string sentence in sat.Split(text)) {
    Console.WriteLine($"[ Sentence: {index++} ]");
    Console.WriteLine(sentence);
    Console.WriteLine();
}
stopwatch.Stop();
Console.WriteLine($"[END] Time : {stopwatch.Elapsed}");
Console.WriteLine("Press Enter to exit...");
Console.ReadLine();