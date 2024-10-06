### **前置準備**

1. **安裝 Python ，如果你已經安裝，就跳過此步驟，我使用的版本是Python 3.10.11，作業系統是Win 11**：
   
   - 確保您已經在 Windows 上安裝了 **Python 3.10.11** 。可以從 [Python 官方網站](https://www.python.org/downloads/windows/) 下載並安裝。

2. **安裝Windows版的Ollama, 本機或其它主機都行，最好是有GPU支援CUDA的最好，AI效能越高的越好 Llama3.2:3b 模型**：

   - 到[Ollama 的官方網站](https://ollama.com/download) 下載安裝和pull模型。
   - 電腦效能不是太高的話，建議不要下載太大的模型，llama3.2 提供1B 、3B共2種參數規模的模型，11B, 90B模型在Ollama沒有
     ，速度還不錯，但中文沒那麼溜，可以使用通義千問 qwen2.5 提供 0.5B 、1.5B 、3B 、7B 、14B 、32B 和 72B 共7種參數規模
   - The default Ollama service is http://127.0.01:11434. The app can run directly. If you install Ollama in A 
     PC and chat-ollama-win in B PC, You need to create a new system Environment Variables as "OLLAMA_HOST=0.0.0.0:11434".
   - **驗證安裝**：
   - 打開 **命令提示字元** 或 **PowerShell**，執行以下命令確認 Ollama 是否安裝成功：
     ```bash
     ollama --version
     ```
   - 應該會顯示 Ollama 的版本資訊。
   
   ```
	ollama pull llama3.2:3b
	ollama pull qwen2.5:7b
	ollama serve 
   ```
   **確認模型運行**：
   - 確保模型已成功運行，並且可以通過 API 訪問。例如：
     ```bash
     ollama list
     ```
   - 應該會顯示正在運行的模型列表。

3. **安裝 FFmpeg（僅限音頻處理）**

   - pydub** 需要 **FFmpeg** 來處理音頻文件。請按照以下步驟在 Windows 上安裝和配置 FFmpeg：

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
   
## **目錄結構**

   **目錄與檔案初始化**

	 為了確保所有必要的目錄和檔案存在，建議運行以下 Python 腳本來初始化目錄結構和檔案：
	### **安裝必要的 Python 套件**

     打開 **命令提示字元（Command Prompt）** 或 **PowerShell**，並執行以下命令來安裝所需的套件： 
	 
```bash
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

### **目錄與檔案說明**

- **`vocabularies/professional_vocab.json`**：專有名詞和習慣用語的詞庫，按照專業分類組織。(還沒用上)

- **`corrections.csv`**：訂正數據文件，儲存用戶對轉寫結果的訂正，供持續訓練使用。(還沒用上)

- **`examples/`**：儲存樣本的目錄。

- **`models/`**：儲存 Whispr 模型的目錄，下載後請放在這。

- **`whisper-ollama-gui.py`**：主 GUI 應用程式，允許用戶選擇音頻文件、設定輸出路徑、管理詞庫、查看和訂正轉寫結果。

- **`initialize.py`**：目錄初始化用，自動建目錄及config.ini。

- **`config.ini`**：參數設定用。

- **`requirements.txt`**：Python 套件依賴列表，藉由 pip install -r requirements.txt 來安裝此程式所需Python套件。

- **`README.md`**：說明文件（概覽、安裝指南）。


### **運行 GUI 應用程式**

確保所有依賴已安裝並且詞庫和訂正數據文件已初始化後，運行 **GUI 程式**：

```
python whisper-ollama-gui.py
```

## **8. 總結**

通過上述步驟和程式碼，您可以在 Windows 環境下建立一個完整的音頻轉文字系統，結合 **Whisper** 進行語音轉文字、**Ollama** 管理後處理模型。


## **進一步的學習資源**
- **Whisper GitHub Repository**：
  - [Whisper by OpenAI](https://github.com/openai/whisper)
- **Ollama 官方文檔**：
  - [Ollama Documentation](https://github.com/ollama/ollama)