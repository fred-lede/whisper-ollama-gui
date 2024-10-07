
# **Sending Whisper Transcripts to LLMs for Post-processing**

## **Prerequisites**

### 1. **Install Python**

If you already have Python installed, skip this step. The version I’m using is Python 3.10.11 on Windows 11.

- Make sure you have **Python 3.10.11** or at least 3.9 installed on Windows. You can download it from the [official Python website](https://www.python.org/downloads/windows/).

### 2. **Install Ollama for Windows**

You can install Ollama either on your local machine or on another host. Ideally, the machine should have a GPU with CUDA support for optimal AI performance.

- Download and install the models from [Ollama’s official website](https://ollama.com/download).
- If your computer’s performance is not high, it’s recommended not to download models that are too large. The `llama3.2` model comes in both 1B and 3B sizes, while `qwen2.5` offers multiple sizes.
- The default Ollama service address is `http://127.0.0.1:11434`.

#### Verifying Installation

1. Open **Command Prompt** or **PowerShell** and run the following command to confirm that Ollama is installed successfully:
   ```bash
   ollama --version
   ```
2. Install the models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:7b
   ollama serve
   ```
3. Check if the models are running:
   ```bash
   ollama list
   ```

### 3. **Install Whisper Model**

- To use **Whisper** for transcription, you need to download the models from OpenAI. You can find the **Whisper** model on the [Whisper GitHub Repository](https://github.com/openai/whisper).
- To download the model that you want and put it into directory "models".

   ```
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    ```
   
	|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
	|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
	|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
	|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
	| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
	| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
	| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
	| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |


### 4. **Install FFmpeg (For Audio Processing Only)**

- **pydub** requires **FFmpeg** to handle audio files. Follow these steps to install and configure FFmpeg on Windows:

#### Installation Steps

1. Download FFmpeg: Go to the [FFmpeg official download page](https://ffmpeg.org/download.html#build-windows) and select the Windows version.
2. Extract FFmpeg to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your system **PATH** environment variable.

#### Verifying Installation

Open **Command Prompt** and run the following command to confirm that FFmpeg is installed successfully:
```bash
ffmpeg -version
```

### 5. **Install PyTorch with CUDA Support**

Install it from [PyTorch’s official page](https://pytorch.org/get-started/locally/).

<img width="607" alt="image" src="https://github.com/user-attachments/assets/0c38bdaf-fc4b-4f75-885d-8b55e4689edf">

## **Directory and File Initialization**

To ensure all necessary directories and files are in place, run the following commands to initialize the directory structure and files:

### Install Required Python Packages

1. Open **Command Prompt** or **PowerShell** and run the following commands:
   ```bash
   git clone https://github.com/fred-lede/chat-ollama-win.git
   cd whisper-ollama
   python -m venv venv

   # Windows PowerShell
   ./venv/Scripts/activate

   # Windows Command Prompt
   venv\Scripts\activate

   pip install -r requirements.txt
   python initialize.py
   ```

### Directory Structure

```bash
whisper-ollama/
├── vocabularies/
│   └── professional_vocab.json
├── output/
│   ├── corrections.csv
│   └── transcript.txt
├── examples/
├── models/
├── whisper-ollama-gui.py
├── initialize.py
├── config.ini
├── requirements.txt
└── README.md
```

### Directory and File Descriptions

- **`vocabularies/professional_vocab.json`**: A vocabulary file that includes technical terms and common phrases, organized by specialization.
- **`corrections.csv`**: Stores the user’s corrections to the transcriptions.
- **`examples/`**: A directory to store example files.
- **`models/`**: Directory for storing Whisper models.
- **`whisper-ollama-gui.py`**: The main GUI application.
- **`initialize.py`**: Directory initialization script.
- **`config.ini`**: Configuration file for parameters.
- **`requirements.txt`**: List of required Python packages.
- **`README.md`**: Documentation.

## **Running the GUI Application**

Run the **GUI application**:
```bash
python whisper-ollama-gui.py
```

## **Summary**

By following these steps, you can set up a complete audio-to-text system on a Windows environment that combines **Whisper** and **Ollama** for post-processing.

## **Further Learning Resources**

- **Whisper GitHub Repository**: [Whisper by OpenAI](https://github.com/openai/whisper)
- **Ollama Official Documentation**: [Ollama Documentation](https://github.com/ollama/ollama)

