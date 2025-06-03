# wtpsplit (C#)

**wtpsplit** is a C# implementation of the sentence and paragraph segmentation logic from the original [wtpsplit](https://github.com/segment-any-text/wtpsplit) project. It uses ONNX Runtime models to split natural language text into meaningful units.

---

## 🔧 Usage

> See also: `Test/Program.cs`

```csharp
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
```

### 📝 Input Example

```text
Today I went to the market I saw a dog it was barking loudly Then I met Sarah she said she was fine but I think she lied we walked for a while and then it started to rain suddenly we ran to the nearest cafe but it was closed we stood there waiting hoping it would stop but it didnt we got wet anyway. After that we went home tired cold wet
```

### 📤 Output Example

```text
[ Sentence: 1 ]
Today I went to the market I saw a dog it was barking loudly

[ Sentence: 2 ]
Then I met Sarah she said she was fine but I think she lied we walked for a while and then it started to rain suddenly we ran to the nearest cafe but it was closed we stood there waiting hoping it would stop but it didnt we got wet anyway.

[ Sentence: 3 ]
After that we went home tired cold wet

[END] Time : 00:00:00.0039188
Press Enter to exit...
```

---

## 📎 Reference

* Original Python version: [segment-any-text/wtpsplit](https://github.com/segment-any-text/wtpsplit)
