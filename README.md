
# **將 Whisper 轉錄出的文字傳送給 LLMs 進行後處理**

中文 | [English](./README-En.md)

## **前置準備**

### 1. **安裝 Python**

如果你已經安裝了 Python，跳過此步驟。我使用的版本是 Python 3.10.11，作業系統是 Windows 11。

- 確保已經在 Windows 上安裝 **Python 3.10.11** 或 3.9 以上版本，可以從 [Python 官方網站](https://www.python.org/downloads/windows/) 下載並安裝。

### 2. **安裝 Windows 版的 Ollama**

可以在本機或其它主機上安裝，最好是有 GPU 支援 CUDA，AI 效能越高越好。

- 到 [Ollama 官方網站](https://ollama.com/download) 下載並安裝模型。
- 如果電腦效能不高，建議不要下載太大的模型，`llama3.2` 提供 1B 和 3B 兩種規模的模型；`qwen2.5` 則提供多種規模。
- 預設的 Ollama 服務地址是 `http://127.0.0.1:11434`。
  
#### 驗證安裝

1. 打開 **命令提示字元** 或 **PowerShell**，執行以下命令確認 Ollama 是否安裝成功：
   ```bash
   ollama --version
   ```
2. 安裝模型：
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:7b
   ollama serve
   ```
3. 確認模型運行：
   ```bash
   ollama list
   ```
### 3. **安裝 Whisper Model**

- 要使用 **Whisper** 進行轉錄，您需要從 OpenAI 下載模型。您可以在 [Whisper GitHub 倉庫](https://github.com/openai/whisper) 上找到 **Whisper** 模型。
- 下載所需的模型並將其放入 "models" 目錄中。

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



### 4. **安裝 FFmpeg（音頻處理）**

- `pydub` 需要 **FFmpeg** 來處理音頻文件。請按照以下步驟在 Windows 上安裝和配置 FFmpeg：

#### 安裝步驟

1. 下載 FFmpeg：前往 [FFmpeg 官方下載頁面](https://ffmpeg.org/download.html#build-windows)，選擇適用於 Windows 的版本。
2. 解壓縮 FFmpeg 至 `C:\ffmpeg`。
3. 將 `C:\ffmpeg\bin` 目錄添加到系統的 **PATH** 環境變數中。

#### 驗證安裝

打開 **命令提示字元**，執行以下命令，確認 FFmpeg 安裝成功：
```bash
ffmpeg -version
```

### 5. **安裝 PyTorch 的 CUDA 支援**

前往 [PyTorch 官方頁面](https://pytorch.org/get-started/locally/) 安裝。

<img width="607" alt="image" src="https://github.com/user-attachments/assets/0c38bdaf-fc4b-4f75-885d-8b55e4689edf">

## **目錄與檔案初始化**

為了確保所有必要的目錄和檔案存在，運行以下命令來初始化目錄結構和檔案：

### 安裝必要的 Python 套件

1. 打開 **命令提示字元（Command Prompt）** 或 **PowerShell**，並執行以下命令：
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

### 目錄結構

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

### 目錄與檔案說明

- **`vocabularies/professional_vocab.json`**：專有名詞和習慣用語的詞庫，按照專業分類組織。
- **`corrections.csv`**：訂正數據文件，儲存用戶對轉寫結果的訂正。
- **`examples/`**：儲存樣本的目錄。
- **`models/`**：儲存 Whisper 模型的目錄。
- **`whisper-ollama-gui.py`**：主 GUI 應用程式。
- **`initialize.py`**：目錄初始化腳本。
- **`config.ini`**：參數設定文件。
- **`requirements.txt`**：Python 套件依賴列表。
- **`README.md`**：說明文件。

## **運行 GUI 應用程式**

運行 **GUI 程式**：
```bash
python whisper-ollama-gui.py
```

## **總結**

通過上述步驟，您可以在 Windows 環境下建立一個完整的音頻轉文字系統，結合 **Whisper** 和 **Ollama** 進行後處理。

## **進一步的學習資源**

- **Whisper GitHub Repository**：[Whisper by OpenAI](https://github.com/openai/whisper)
- **Ollama 官方文檔**：[Ollama Documentation](https://github.com/ollama/ollama)

