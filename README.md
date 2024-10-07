# **將Whisper轉錄出的文字傳送給LLMs進行後處理**

## **前置準備**

1. **安裝 Python ，如果你已經安裝，就跳過此步驟，我使用的版本是Python 3.10.11，作業系統是Win 11：**
   - 確保您已經在 Windows 上安裝了 **Python 3.10.11** 或3.9以上版本 。可以從 [Python 官方網站](https://www.python.org/downloads/windows/) 下載並安裝。
2. **安裝Windows版的Ollama, 本機或其它主機都行，最好是有GPU支援CUDA的最好，AI效能越高的越好：**
   - 到[Ollama 的官方網站](https://ollama.com/download) 下載安裝和pull模型。
   - 電腦效能不是太高的話，建議不要下載太大的模型，llama3.2提供1B 、3B共2種參數規模的模型，11B, 90B模型在Ollama沒有
     ，llama3.2:3b速度還不錯，但中文沒那麼溜，可以使用通義千問qwen2.5，有提供 0.5B 、1.5B 、3B 、7B 、14B 、32B 和 72B 共7種參數規模
   - 預設的Ollama服務是http://127.0.01:11434。應用程式可以直接運行。如果在一台PC上安裝了Ollama，在另一台PC上安裝了whisper-ollama-gui，
     則Ollama服務主機需要創建一個新的系統環境變數為“Ollama”_主機= 0.0.0.0:11434”。
   - 驗證安裝：
     打開 **命令提示字元** 或 **PowerShell**，執行以下命令確認 Ollama 是否安裝成功：
     ```
     ollama --version
     ```
     應該會顯示 Ollama 的版本資訊。
     ```
     ollama pull llama3.2:3b
     ollama pull qwen2.5:7b
     ollama serve 
     ```
     確認模型運行：
     確保模型已成功運行，並且可以通過 API 訪問。例如：
     ```
     ollama list
     ```
     應該會顯示正在運行的模型列表，表示你的Ollama已正常運作待命中了。

3. **安裝 FFmpeg（僅限音頻處理）**

   - pydub** 套件需要 **FFmpeg** 來處理音頻文件。請按照以下步驟在 Windows 上安裝和配置 FFmpeg：

   - 下載 FFmpeg：
     前往 [FFmpeg 官方下載頁面](https://ffmpeg.org/download.html#build-windows)。
     選擇適用於 Windows 的靜態編譯版本，例如 [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 提供的版本。

   - 解壓縮 FFmpeg：
     將下載的壓縮包解壓縮到一個目錄，例如 `C:\ffmpeg`.

   - 配置環境變數：
   - 將 FFmpeg 的 `bin` 目錄添加到系統的 **PATH** 環境變數中。
     右鍵點擊 **此電腦** > **屬性** > **高級系統設置** > **環境變數**。
     在 **系統變數** 中找到 `Path`，選擇 **編輯**。
     點擊 **新建**，添加 `C:\ffmpeg\bin`（根據實際解壓路徑調整）。
     確認所有對話框以保存變更。

   - 驗證安裝：
     打開 **命令提示字元**，執行 `ffmpeg -version`，應顯示 FFmpeg 的版本資訊。

4. **安裝PyTorch’s CUDA support**
   - 前往 [PyTorch 官方頁面](https://pytorch.org/get-started/locally/)。
















   <img width="607" alt="image" src="https://github.com/user-attachments/assets/0c38bdaf-fc4b-4f75-885d-8b55e4689edf">

   ```
   C:\python
   Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
	Type "help", "copyright", "credits" or "license" for more information.
	>>> import torch
	>>> torch.cuda.is_available()
	True
	>>>
    ```   
	True表示你的CUDA已可使用
5. **下載Whisper所需要的模型**
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

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3` that offers faster transcription speed with a minimal degradation in accuracy.
   
## **目錄與檔案初始化**
   為了確保所有必要的目錄和檔案存在，建議運行以下 Python 腳本來初始化目錄結構和檔案：
   **安裝必要的 Python 套件**
   打開 **命令提示字元（Command Prompt）** 或 **PowerShell**，並執行以下命令來安裝所需的套件： 
```
git clone https://github.com/fred-lede/chat-ollama-win.git
cd whisper-ollama
python -m venv venv

#Windows PowerShell
./venv/Scripts/activate

#Windows Command
venv\Scripts\activate

pip install -r requirements.txt

python initialize.py
```

```
whisper-ollama/
├── vocabularies/
│   └── professional_vocab.json
├── output/
│   ├── corrections.csv
│   └── transcript.txt
│── examoles/
├── models/
├── whisper-ollama-gui.py
├── initialize.py
│── config.ini
├── requirements.txt
└── README.md
```

  **目錄與檔案說明**

- **`vocabularies/professional_vocab.json`**：專有名詞和習慣用語的詞庫，按照專業分類組織。(還沒用上)

- **`corrections.csv`**：訂正數據文件，儲存用戶對轉寫結果的訂正，供持續訓練使用。(還沒用上)

- **`examples/`**：儲存樣本的目錄。

- **`models/`**：儲存 Whispr 模型的目錄，下載後請放在這。

- **`whisper-ollama-gui.py`**：主 GUI 應用程式，允許用戶選擇音頻文件、設定輸出路徑、管理詞庫、查看和訂正轉寫結果。

- **`initialize.py`**：目錄初始化用，自動建目錄及config.ini。

- **`config.ini`**：參數設定用。

- **`requirements.txt`**：Python 套件依賴列表，藉由 pip install -r requirements.txt 來安裝此程式所需Python套件。

- **`README.md`**：說明文件（概覽、安裝指南）。

## **運行 GUI 應用程式**

確保所有依賴已安裝並且詞庫和訂正數據文件已初始化後，運行 **GUI 程式**：

```
python whisper-ollama-gui.py
```

## **總結：**

通過上述步驟和程式碼，您可以在 Windows 環境下建立一個完整的音頻轉文字系統，結合 **Whisper** 進行語音轉文字、**Ollama** 管理後處理模型。


## **進一步的學習資源**
- **Whisper GitHub Repository**：
  - [Whisper by OpenAI](https://github.com/openai/whisper)
- **Ollama 官方文檔**：
  - [Ollama Documentation](https://github.com/ollama/ollama)
