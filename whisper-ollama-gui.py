import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import whisper
import torch
import os
import threading
import librosa
import soundfile as sf
from pydub import AudioSegment
import pandas as pd
import requests
import json
import configparser
import time

# 讀取配置檔案
config = configparser.ConfigParser()
config_file = 'config.ini'

if not os.path.exists(config_file):
    # 如果配置檔案不存在，創建一個默認配置檔案
    current_directory = os.getcwd()
    #print(f"當前運行的目錄路徑是：{current_directory}")
    
    default_config = {
        'Server': {
            'Address': '127.0.0.1',
            'Port': '11434'
        },
        'Models': {
            'ModelsDirectory': current_directory+'\\models'
        },
        'Output': {
            'OutputDirectory': current_directory+'\\output'
        },
        'Whisper': {
            'SupportedLanguages': 'Chinese, English, Thai, Japanese, Korean, Malayalam, Vietnamese, Indonesian'
        }
    }
    with open(config_file, 'w') as f:
        config.read_dict(default_config)
        config.write(f)
else:
    config.read(config_file)

# 配置變數
OLLAMA_SERVICE_URL = config['Server']['Address']
OLLAMA_SERVICE_PORT = config['Server']['Port']
OLLAMA_API_URL = f"http://{OLLAMA_SERVICE_URL}:{OLLAMA_SERVICE_PORT}"
MODELS_DIRECTORY = config['Models']['ModelsDirectory']
OUTPUT_DIRECTORY = config['Output']['OutputDirectory']

# 讀取支持的語言
SUPPORTED_LANGUAGES = config['Whisper']['SupportedLanguages'].split(", ")

# 設定裝置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 載入詞庫
def load_vocabularies():
    vocab_file = "vocabularies/professional_vocab.json"
    if not os.path.exists(vocab_file):
        # 創建一個空詞庫文件
        empty_vocab = {
            "電動車": {
                "專有名詞": [],
                "習慣用語": []
            },
            "醫療": {
                "專有名詞": [],
                "習慣用語": []
            },
            "法律": {
                "專有名詞": [],
                "習慣用語": []
            },
            "技術": {
                "專有名詞": [],
                "習慣用語": []
            }
        }
        os.makedirs("vocabularies", exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(empty_vocab, f, ensure_ascii=False, indent=4)
    
    with open(vocab_file, "r", encoding="utf-8") as f:
        return json.load(f)

vocabularies = load_vocabularies()

def get_custom_vocab():
    custom_vocab = []
    for category, types in vocabularies.items():
        custom_vocab.extend(types.get("專有名詞", []))
        custom_vocab.extend(types.get("習慣用語", []))
    return custom_vocab

# 擴充 Whisper 的詞彙表（如果需要）
def extend_whisper_vocab(model, custom_vocab):
    if custom_vocab:
        # Whisper 的 tokenizer 目前不支援動態添加詞彙
        # 如果需要自訂詞彙，可能需要重新訓練模型或使用其他方法
        # 此處僅提醒用戶
        center_custom_messagebox("詞彙擴充", "Whisper 的 tokenizer 不支援動態添加詞彙。請確保您的詞彙已包含在模型中。")
# 掃描指定目錄中的 Whisper 模型檔案，並返回模型名稱列表
def load_whisper_model_list(model_directory=MODELS_DIRECTORY):
    if not os.path.exists(model_directory):
        center_custom_messagebox("模型目錄缺失", f"模型目錄 {model_directory} 未找到。")
        return []
    
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pt')]
    model_names = [os.path.splitext(f)[0] for f in model_files]
    return model_names

# 載入選定的 Whisper 模型
def load_selected_whisper_model(selected_model):
    model_path = os.path.join(MODELS_DIRECTORY, f"{selected_model}.pt")
    if not os.path.exists(model_path):
        center_custom_messagebox("模型缺失", f"模型檔案 {model_path} 未找到。")
        return None
    try:
        model = whisper.load_model(model_path).to(DEVICE)
        custom_vocab = get_custom_vocab()
        # extend_whisper_vocab(model, custom_vocab)  # Whisper 的 tokenizer 不支援動態添加詞彙
        return model
    except Exception as e:
        center_custom_messagebox("模型載入錯誤", f"無法載入模型 {selected_model}: {e}")
        return None

whisper_models = load_whisper_model_list()

# 訓練數據存儲路徑
CORRECTION_DATA_PATH = os.path.join(OUTPUT_DIRECTORY, "corrections.csv")

# 初始化或創建訂正數據文件
if not os.path.exists(CORRECTION_DATA_PATH):
    df = pd.DataFrame(columns=["input", "output"])
    df.to_csv(CORRECTION_DATA_PATH, index=False)

# 函數：從 Ollama 服務器獲取可用的模型列表
def fetch_ollama_models(server_address=OLLAMA_API_URL):
    try:
        response = requests.get(f"{server_address}/v1/models")
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("object") == "list" and "data" in data:
            models = data["data"]
            return [model["id"] for model in models]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected response format")
    except requests.RequestException as e:
        center_custom_messagebox("Error", f"Failed to retrieve models: {e}")
        print(f"Error details: {e}")
    except ValueError as ve:
        center_custom_messagebox("Error", str(ve))
        print(f"ValueError: {ve}")
    return []

def preprocess_audio(audio_path):
    """
    預處理音頻：轉換為單聲道和16kHz的WAV格式。
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        temp_path = "temp_audio.wav"
        audio.export(temp_path, format="wav")

        y, sr = librosa.load(temp_path, sr=16000)
        sf.write(temp_path, y, sr)
        return temp_path
    except Exception as e:
        center_custom_messagebox("預處理錯誤", f"音頻預處理失敗: {e}")
        return None

def transcribe_audio(model, audio_path, whisper_params, progress_var):
    """
    使用 Whisper 將音頻轉換為文字，根據提供的參數。
    """
    
    try:
        progress_var.set(10)
        result = model.transcribe(audio_path, **whisper_params)
        progress_var.set(40)
        stop_whisper_flickering()  # 子程序結束時停止閃爍
        return result['text']
    except Exception as e:
        center_custom_messagebox("轉寫錯誤", f"Whisper 轉寫失敗: {e}")
        stop_whisper_flickering()
        return None

def post_process_text_with_ollama(server_address, selected_post_model, text, progress_var, selected_question_var):
    """
    使用 Ollama 的模型進行文字後處理，考慮專有名詞和習慣用語。
    """
    start_ollama_flickering_in_subprocess()  # 子程序運行，開始閃爍 
    
    try:
        # 根據 Ollama 的實際 API 端點調整
        ollama_api_url = f"{server_address}/v1/chat/completions"  # 請根據實際 API 端點調整
        headers = {
            "Content-Type": "application/json"
        }
        
        # center_custom_messagebox("Debug", ollama_api_url+"--"+selected_question_var.get())
        # 訂正錯字, 訂正錯字+潤飾, 中文翻譯, 英文翻譯
        if selected_question_var.get() == "訂正錯字":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文訂正以下文字的錯別字，請分成二個部分回答，第一個部分先條列說明修改的地方，第二個部分則說明訂正後的全文：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "訂正錯字+潤飾":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文訂正錯字及潤飾以下文字，提升其準確性和可讀性，請分成二個部分回答，第一個部分先條列說明訂正錯字及潤飾的地方，第二個部分則說明訂正後的全文：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "會議摘要":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文訂正錯字並以條列式完成一份專業的會議重點摘要報告，請分成二個部分回答，第一個部分先說明修改的地方，第二個部分則說明你完成的條列式會議摘要報告：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "會議紀錄":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文訂正錯字並完成一份專業的會議紀錄，請分成二個部分回答，第一個部分先說明修改的地方，第二個部分則顯示你完成的會議紀錄：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "重點摘要":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文訂正錯字及潤飾再以條列式完成重點摘要，請分成二個部分回答，第一個部分先說明修改的地方，第二個部分則顯示你完成的條列式重點摘要：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "一行一句":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請根據前後文修正以下文字的錯別字，並加以潤飾和斷句。請提供兩種回答：第一種回答詳述修改的地方；第二種回答以一行一句的方式呈現。\n\n文字如下：\n" + text + "\n請根據上述要求進行修改。"
                    }
                ]
            }   
        elif selected_question_var.get() == "中文翻譯":
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請用中文翻譯以下文字並以二個版本回答，第一個版本為直接翻譯輸出，第二個版本為翻譯、潤飾再輸出：\n" + text
                    }
                ]
            }
        elif selected_question_var.get() == "英文翻譯":   
            payload = {
                "model": selected_post_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "請用英文翻譯以下文字並以二個版本回答，第一個版本為直接翻譯輸出，第二個版本為翻譯、潤飾再輸出：\n" + text
                    }
                ]
            }    
        else:
            center_custom_messagebox("後處理錯誤", "後處理類型無效選項")
            stop_ollama_flickering()
            return text
                
        progress_var.set(50)
        
        # 發送 POST 請求
        response = requests.post(ollama_api_url, headers=headers, json=payload)
        
        # 調試：打印響應狀態碼和內容
        print(f"Response Status Code: {response.status_code}")
        # print(f"Response Content: {response.text}")
        
        # 停止閃爍
        stop_ollama_flickering()  # 子程序結束時停止閃爍
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                # 根據 Ollama 的回應結構進行調整
                # 假設回應格式為 { "choices": [{"message": {"content": "processed_text"}}] }
                processed_text = response_json['choices'][0]['message']['content']
                progress_var.set(80)
                return processed_text
            except json.JSONDecodeError:
                center_custom_messagebox("後處理錯誤", "無法解析 Ollama 的響應：無效的 JSON 格式")
                return text
            except KeyError:
                center_custom_messagebox("後處理錯誤", f"回應格式不正確：{response_json}")
                return text
        else:
            center_custom_messagebox("後處理錯誤", f"Ollama 後處理請求失敗: {response.status_code} {response.text}")
            return text
    except Exception as e:
        center_custom_messagebox("後處理錯誤", f"Ollama 後處理失敗: {e}")
        stop_ollama_flickering()  # 子程序結束時停止閃爍
        return text

def process_audio(file_path, text_area_raw, text_area_edit, selected_post_model, whisper_params, output_path, output_filename, ollama_server, progress_var, selected_question_var):
    """
    完整的音頻處理流程：預處理、轉寫、後處理，並保存結果。
    """
    progress_var.set(0)
    # 預處理音頻
    preprocessed_path = preprocess_audio(file_path)
    if not preprocessed_path:
        progress_var.set(0)
        return

    # 轉寫音頻
    whisper_model = load_selected_whisper_model(selected_whisper_model_var.get())
    if not whisper_model:
        progress_var.set(0)
        return
    text = transcribe_audio(whisper_model, preprocessed_path, whisper_params, progress_var)
    if not text:
        progress_var.set(0)
        return

    # 更新原始文本框
    text_area_raw.config(state=tk.NORMAL)  # 啟用文本區域
    text_area_raw.delete("1.0", tk.END)
    text_area_raw.insert(tk.END, text)
    text_area_raw.config(state=tk.DISABLED)  # 設置為唯讀
    

    # 後處理文字
    processed_text = post_process_text_with_ollama(ollama_server, selected_post_model, text, progress_var, selected_question_var)

    # 更新編輯文本框
    text_area_edit.delete("1.0", tk.END)
    text_area_edit.insert(tk.END, processed_text)
    

    # 顯示進度
    progress_var.set(100)

    # 保存結果到指定路徑
    try:
        with open(os.path.join(output_path, output_filename), "w", encoding="utf-8") as f:
            f.write(processed_text)
    except Exception as e:
        center_custom_messagebox("儲存錯誤", f"無法儲存轉寫結果: {e}")

    # 清理臨時文件
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

def select_file(text_area_raw, text_area_edit, post_model_selector, whisper_params, output_path_var, output_filename_var, ollama_server_var, progress_var, selected_question_var):
    """
    開啟檔案選擇對話框，並開始處理選擇的音頻檔案。
    """
    file_path = filedialog.askopenfilename(
        title="選擇音頻檔案",
        filetypes=[("音頻檔案", "*.mp3 *.wav *.m4a *.flac"), ("視頻檔案", "*.mp4 *.mpeg *.m4a"), ("所有檔案", "*.*")]
    )
    if file_path:
        selected_post_model = post_model_selector.get()
        if not selected_post_model:
            center_custom_messagebox("未選擇後處理模型", "請在下拉選單中選擇一個後處理模型。")
            return
        output_path = output_path_var.get()
        output_filename = output_filename_var.get()
        if not output_path:
            center_custom_messagebox("未選擇輸出路徑", "請選擇轉寫結果的儲存目錄。")
            return
        if not output_filename:
            center_custom_messagebox("未輸入檔名", "請輸入轉寫結果的檔名。")
            return
        ollama_server = ollama_server_var.get()
        if not ollama_server:
            center_custom_messagebox("未設定 Ollama 服務器", "請在配置檔案中設定 Ollama 服務器地址和端口。")
            return
        
        # print(whisper_params)
        # 設定 Whisper 參數
        whisper_params_updated = {
            "language": whisper_params["language"],
            "task": whisper_params["task"],
            "verbose": True,      # True:顯示, False:不顯示細節
            "temperature": whisper_params["temperature"],
            "best_of": whisper_params["best_of"],
            "beam_size": whisper_params["beam_size"],
        }
        print(whisper_params_updated)
        
        start_whisper_flickering_in_subprocess()  # 子程序運行，開始閃爍        
        # 開啟新執行緒以避免阻塞 GUI
        threading.Thread(target=process_audio, args=(file_path, text_area_raw, text_area_edit, selected_post_model, whisper_params_updated, output_path, output_filename, ollama_server, progress_var, selected_question_var), daemon=True).start()

def save_correction(text_area_edit):
    """
    儲存訂正後的文本作為訓練數據。
    """
    original_text = text_area_edit.get("1.0", tk.END).strip()
    if not original_text:
        center_custom_messagebox("無內容", "轉寫結果為空，無法儲存。")
        return

    # 彈出一個新的窗口讓用戶確認或修改訂正
    def confirm_save():
        corrected_text = edit_text_area.get("1.0", tk.END).strip()
        if not corrected_text:
            center_custom_messagebox("無內容", "訂正後的文本為空，無法儲存。")
            return
        # 保存到 CSV
        try:
            new_row = pd.DataFrame([{"input": original_text, "output": corrected_text}])
            new_row.to_csv(CORRECTION_DATA_PATH, mode='a', index=False, header=False, encoding='utf-8')
            center_custom_messagebox("成功", "訂正結果已儲存。")
            confirm_window.destroy()
            root.attributes("-disabled", False)  # 確保在保存後啟用主視窗
        except Exception as e:
            center_custom_messagebox("儲存錯誤", f"無法儲存訂正結果: {e}")

    confirm_window = tk.Toplevel(root)
    confirm_window.title("訂正文本")
    confirm_window.geometry("600x500+500+300")  # 設定子視窗大小與位置
    
    
    
    # 禁用主視窗
    root.attributes("-disabled", True)
    
    # 當子視窗關閉時，重新啟用主視窗
    def on_close():
        root.attributes("-disabled", False)
        confirm_window.destroy()
        
    confirm_window.protocol("WM_DELETE_WINDOW", on_close)
    
    global edit_text_area

    tk.Label(confirm_window, text="請檢查並修改訂正後的文本：").pack(pady=5)
    edit_text_area = scrolledtext.ScrolledText(confirm_window, wrap=tk.WORD, width=100, height=30, undo=True)
    edit_text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    edit_text_area.insert(tk.END, original_text)
    
    # 創建右鍵選單
    right_click_menu_edit = tk.Menu(root, tearoff=0)
    right_click_menu_edit.add_command(label="複製(C)", command=lambda: copy_to_clipboard(edit_text_area))
    right_click_menu_edit.add_command(label="剪下(T)", command=lambda: edit_text_area.event_generate("<<Cut>>"))
    right_click_menu_edit.add_command(label="貼上(P)", command=lambda: edit_text_area.event_generate("<<Paste>>"))
    right_click_menu_edit.add_command(label="刪除(D)", command=lambda: delete_selection(edit_text_area))
    right_click_menu_edit.add_separator()
    right_click_menu_edit.add_command(label="全選(A)", command=lambda: select_all(edit_text_area))
    right_click_menu_edit.add_separator()
    right_click_menu_edit.add_command(label="復原(Z)", command=lambda: undo_action(edit_text_area))
    right_click_menu_edit.add_command(label="重做(Y)", command=lambda: redo_action(edit_text_area))


    # 綁定右鍵點擊事件
    edit_text_area.bind("<Button-3>", lambda event: popup(event, edit_text_area, right_click_menu_edit))

    save_button = tk.Button(confirm_window, text="儲存訂正", command=confirm_save, width=10, height=1)
    save_button.pack(pady=10)
    
    confirm_window.wait_window(confirm_window)  # 確保等待子視窗關閉後，恢復主視窗的狀態


def manage_vocabulary():
    """
    打開詞庫管理窗口，允許用戶添加、刪除和分類詞彙。
    """
    manage_window = tk.Toplevel(root)
    manage_window.title("詞庫管理")
    manage_window.geometry("600x500+500+300")  # 設定子視窗大小與位置
    
    # 禁用主視窗
    root.attributes("-disabled", True)
    
    # 當子視窗關閉時，重新啟用主視窗
    def on_close():
        root.attributes("-disabled", False)
        manage_window.destroy()
        
    manage_window.protocol("WM_DELETE_WINDOW", on_close)
    
    # 專業分類選擇
    tk.Label(manage_window, text="選擇專業分類：").pack(pady=5)
    category_var = tk.StringVar()
    category_options = list(vocabularies.keys())
    category_menu = ttk.Combobox(manage_window, textvariable=category_var, values=category_options, state="readonly")
    category_menu.pack(pady=5)
    
    # 詞彙類型選擇
    tk.Label(manage_window, text="選擇詞彙類型：").pack(pady=5)
    type_var = tk.StringVar()
    type_options = ["專有名詞", "習慣用語"]
    type_menu = ttk.Combobox(manage_window, textvariable=type_var, values=type_options, state="readonly")
    type_menu.pack(pady=5)
    
    # 詞彙輸入
    tk.Label(manage_window, text="輸入詞彙：").pack(pady=5)
    vocab_entry = tk.Entry(manage_window, width=50)
    vocab_entry.pack(pady=5)

    # 現有詞彙顯示
    tk.Label(manage_window, text="現有詞彙：").pack(pady=5)
    existing_vocab_text = scrolledtext.ScrolledText(manage_window, wrap=tk.WORD, width=70, height=10, state='disabled')
    existing_vocab_text.pack(pady=5)
    
    def update_existing_vocab(event=None):
        category = category_var.get()
        vocab_type = type_var.get()
        if category and vocab_type:
            existing_vocab = vocabularies[category][vocab_type]
            existing_vocab_text.config(state='normal')
            existing_vocab_text.delete("1.0", tk.END)
            existing_vocab_text.insert(tk.END, "\n".join(existing_vocab))
            existing_vocab_text.config(state='disabled')
    
    category_menu.bind("<<ComboboxSelected>>", update_existing_vocab)
    type_menu.bind("<<ComboboxSelected>>", update_existing_vocab)
    
    # 添加詞彙按鈕
    def add_vocab():
        category = category_var.get()
        vocab_type = type_var.get()
        vocab = vocab_entry.get().strip()
        if not category or not vocab_type or not vocab:
            center_custom_messagebox("輸入不完整", "請填寫所有欄位。")
            return
        if vocab in vocabularies[category][vocab_type]:
            center_custom_messagebox("詞彙已存在", f"詞彙「{vocab}」已在 {category} 的 {vocab_type} 中。")
            return
        vocabularies[category][vocab_type].append(vocab)
        # 更新詞庫文件
        with open("vocabularies/professional_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocabularies, f, ensure_ascii=False, indent=4)
        center_custom_messagebox("成功", f"已添加詞彙：{vocab}")
        vocab_entry.delete(0, tk.END)
        update_existing_vocab()
        # 注意：OpenAI Whisper 的 tokenizer 不支援動態添加詞彙
        # 如需擴充詞彙，可能需要重新訓練模型或使用其他方法
    
    add_button = tk.Button(manage_window, text="添加詞彙", command=add_vocab, width=10, height=1)
    add_button.pack(pady=10)
    
    # 刪除詞彙功能
    def delete_vocab():
        category = category_var.get()
        vocab_type = type_var.get()
        vocab = vocab_entry.get().strip()
        if not category or not vocab_type or not vocab:
            center_custom_messagebox("輸入不完整", "請填寫所有欄位。")
            return
        if vocab not in vocabularies[category][vocab_type]:
            center_custom_messagebox("詞彙不存在", f"詞彙「{vocab}」不在 {category} 的 {vocab_type} 中。")
            return
        vocabularies[category][vocab_type].remove(vocab)
        # 更新詞庫文件
        with open("vocabularies/professional_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocabularies, f, ensure_ascii=False, indent=4)
        center_custom_messagebox("成功", f"已刪除詞彙：{vocab}")
        vocab_entry.delete(0, tk.END)
        update_existing_vocab()
    
    delete_button = tk.Button(manage_window, text="刪除詞彙", command=delete_vocab, width=10, height=1)
    delete_button.pack(pady=5)
    
    return root
    
def choose_output_path(output_path_var):
    """
    開啟目錄選擇對話框，讓用戶選擇轉寫結果的儲存目錄。
    """
    path = filedialog.askdirectory(title="選擇輸出目錄")
    if path:
        output_path_var.set(path)
        
# --- 閃爍功能部分 ---
stop_whisper_flicker = False  # 停止閃爍的標誌位
stop_ollama_flicker = False   # 停止閃爍的標誌位
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']

def toggle_whisper_border_color(frame):
    def update_color():
        global stop_whisper_flicker
        while not stop_whisper_flicker:
            for color in colors:
                if stop_whisper_flicker:
                    break
                frame.config(highlightbackground=color, highlightcolor=color, highlightthickness=1)
                frame.update_idletasks()
                time.sleep(0.5)  # 控制閃爍速度
    threading.Thread(target=update_color, daemon=True).start()

def stop_whisper_flickering():
    global stop_whisper_flicker
    stop_whisper_flicker = True
    whisper_frame.config(highlightbackground="SystemButtonFace", highlightcolor="SystemButtonFace", highlightthickness=1)

def toggle_ollama_border_color(frame):
    def update_color():
        global stop_ollama_flicker
        while not stop_ollama_flicker:
            for color in colors:
                if stop_ollama_flicker:
                    break
                frame.config(highlightbackground=color, highlightcolor=color, highlightthickness=1)
                frame.update_idletasks()
                time.sleep(0.5)  # 控制閃爍速度
    threading.Thread(target=update_color, daemon=True).start()

def stop_ollama_flickering():
    global stop_ollama_flicker
    stop_ollama_flicker = True
    ollama_frame.config(highlightbackground="SystemButtonFace", highlightcolor="SystemButtonFace", highlightthickness=1)

def start_whisper_flickering_in_subprocess():
    global stop_whisper_flicker
    stop_whisper_flicker = False
    toggle_whisper_border_color(whisper_frame)

def start_ollama_flickering_in_subprocess():
    global stop_ollama_flicker
    stop_ollama_flicker = False
    toggle_ollama_border_color(ollama_frame)

def center_custom_messagebox(title, message):
    global root
    root.update_idletasks()
    
    # 獲取主視窗的位置和大小
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    root_height = root.winfo_height()

    # 計算主視窗的中心點
    center_x = root_x + root_width // 2
    center_y = root_y + root_height // 2

    # 計算自定義對話框的位置
    messagebox_width = 300
    messagebox_height = 150
    position_x = center_x - messagebox_width // 2
    position_y = center_y - messagebox_height // 2

    # 創建自定義對話框
    messagebox_window = tk.Toplevel(root)
    messagebox_window.title(title)
    messagebox_window.geometry(f"{messagebox_width}x{messagebox_height}+{position_x}+{position_y}")
    messagebox_window.transient(root)  # 設置為主視窗的子窗口
    messagebox_window.grab_set()  # 阻止主視窗操作

    # 添加訊息標籤
    label = tk.Label(messagebox_window, text=message, wraplength=250)
    label.pack(pady=20)

    # 添加確認按鈕
    ok_button = ttk.Button(messagebox_window, text="確認", command=messagebox_window.destroy)
    ok_button.pack(pady=10)

    # 等待對話框關閉
    root.wait_window(messagebox_window)

def popup(event, text_widget, menu):
    try:
        menu.tk_popup(event.x_root, event.y_root)
    finally:
        menu.grab_release()

def copy_to_clipboard(text_widget):
    try:
        # 嘗試獲取選取的文字
        text = text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        root.clipboard_clear()
        root.clipboard_append(text)
    except tk.TclError:
        pass  # 沒有選取的文字時，不做任何操作

def select_all(text_widget):
    text_widget.tag_add(tk.SEL, "1.0", tk.END)
    text_widget.mark_set(tk.INSERT, "1.0")
    text_widget.see(tk.INSERT)

def delete_selection(text_widget):
    try:
        text_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
    except tk.TclError:
        pass  # 沒有選取的文字時，不做任何操作

def undo_action(text_widget):
    try:
        text_widget.edit_undo()
    except tk.TclError:
        pass  # 沒有可復原的操作

def redo_action(text_widget):
    try:
        text_widget.edit_redo()
    except tk.TclError:
        pass  # 沒有可重做的操作
        
def clear_text():
    text_area_raw.config(state=tk.NORMAL)
    text_area_raw.delete('1.0', tk.END)
    text_area_raw.config(state=tk.DISABLED)
    text_area_edit.delete('1.0', tk.END)
    
def validate_inputs():
    # 檢查 temperature 的範圍
    if not (0 <= temperature_var.get() <= 1):
        messagebox.showerror("錯誤", "Temperature 的範圍應該在 0 到 1 之間")
        return False

    # 檢查 best_of 的範圍
    if not (1 <= best_of_var.get() <= 10):
        messagebox.showerror("錯誤", "Best of 的範圍應該在 1 到 10 之間")
        return False

    # 檢查 beam_size 的範圍
    if not (1 <= beam_size_var.get() <= 10):
        messagebox.showerror("錯誤", "Beam size 的範圍應該在 1 到 10 之間")
        return False

    return True
           

# --- 圖形使用者介面建立 ---
def create_gui():
    """
    創建圖形使用者介面。
    """
    global root, whisper_frame, ollama_frame, text_area_raw, text_area_edit, temperature_var, best_of_var, beam_size_var
    root = tk.Tk()
    root.title("影音檔轉文字工具")

    # 設定視窗大小
    root.geometry("1400x900+100+100")  # 設定主視窗大小與位置

    # --- Whisper 參數設定區 ---
    whisper_frame = tk.LabelFrame(root, text="Whisper 參數設定", padx=10, pady=10)
    whisper_frame.pack(padx=10, pady=10, fill="x")   
    
    # 模型選擇
    tk.Label(whisper_frame, text="選擇 Whisper 模型：").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    global selected_whisper_model_var
    selected_whisper_model_var = tk.StringVar()
    model_options = whisper_models
    model_menu = ttk.Combobox(whisper_frame, textvariable=selected_whisper_model_var, values=model_options, state="readonly")
    model_menu.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    if model_options:
        selected_whisper_model_var.set(model_options[0])  # 設定預設選項
    # 顯示GPU or CPU    
    tk.Label(whisper_frame, text="Device：" + DEVICE).grid(row=0, column=2, sticky="w", padx=5, pady=5)
    
    # Temperature 輸入框
    tk.Label(whisper_frame, text="Temperature：").grid(row=0, column=4, sticky="e", padx=5, pady=5)
    temperature_var = tk.DoubleVar(value=0)  # 預設值 0
    temperature_entry = tk.Entry(whisper_frame, textvariable=temperature_var, width=10)
    temperature_entry.grid(row=0, column=5, sticky="w", padx=5, pady=5)
    tk.Label(whisper_frame, text="（範圍：0 ~ 1.0, 較低的值會使模型生成更確定、重複性高的結果，較高的值會使結果更加多樣性和隨機性。）").grid(row=0, column=6, columnspan=2, sticky="w", padx=5, pady=5)      
    
     # 語言選擇
    tk.Label(whisper_frame, text="選擇輸入音頻的語言：").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    selected_language_var = tk.StringVar(value=SUPPORTED_LANGUAGES[0] if SUPPORTED_LANGUAGES else "無語言可用")
    language_menu = ttk.Combobox(whisper_frame, textvariable=selected_language_var, values=SUPPORTED_LANGUAGES, state="readonly")
    language_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    tk.Label(whisper_frame, text="（語言請於 config.ini 內增減）").grid(row=1, column=2, sticky="w")
    
    # Best of 輸入框
    tk.Label(whisper_frame, text="Best of：").grid(row=1, column=4, sticky="e", padx=5, pady=5)
    best_of_var = tk.IntVar(value=1)  # 預設值 1
    best_of_entry = tk.Entry(whisper_frame, textvariable=best_of_var, width=10)
    best_of_entry.grid(row=1, column=5, sticky="w", padx=5, pady=5)
    tk.Label(whisper_frame, text="（範圍：1 ~ 10, 指定生成多個候選結果後選擇最佳的一個，但會增加處理時間。）").grid(row=1, column=6, columnspan=2, sticky="w", padx=5, pady=5)  
    
    # 任務類型選擇
    tk.Label(whisper_frame, text="選擇任務類型：").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    task_var = tk.StringVar(value="transcribe")
    task_options = ["transcribe", "translate"]
    task_menu = ttk.Combobox(whisper_frame, textvariable=task_var, values=task_options, state="readonly")
    task_menu.grid(row=2, column=1, sticky="w", padx=5, pady=5)
    tk.Label(whisper_frame, text="（轉錄 transcribe、翻譯 translate）").grid(row=2, column=2, sticky="w")
        
    # Beam size 輸入框
    tk.Label(whisper_frame, text="Beam size：").grid(row=2, column=4, sticky="e", padx=5, pady=5)
    beam_size_var = tk.IntVar(value=5)  # 預設值 5
    beam_size_entry = tk.Entry(whisper_frame, textvariable=beam_size_var, width=10)
    beam_size_entry.grid(row=2, column=5, sticky="w", padx=5, pady=5)
    tk.Label(whisper_frame, text="（範圍：1 ~ 10, 較大的 Beam Size 通常能提高轉錄的準確性，但會增加計算資源的消耗和處理時間。）").grid(row=2, column=6, columnspan=2, sticky="w", padx=5, pady=5)  
    
    
    # --- Ollama 服務器設定區 ---
    ollama_frame = tk.LabelFrame(root, text="Ollama 服務器設定", padx=10, pady=10)
    ollama_frame.pack(padx=10, pady=10, fill="x")

    tk.Label(ollama_frame, text="Ollama 服務器地址：").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ollama_server_var = tk.StringVar(value=f"http://{OLLAMA_SERVICE_URL}:{OLLAMA_SERVICE_PORT}")
    ollama_server_entry = tk.Entry(ollama_frame, textvariable=ollama_server_var, width=30)
    ollama_server_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    # 後處理模型選擇
    tk.Label(ollama_frame, text="選擇後處理模型：").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    post_model_selector = ttk.Combobox(ollama_frame, state="readonly", width=30)
    post_model_selector.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    def update_post_models():
        server_address = ollama_server_var.get()
        models = fetch_ollama_models(server_address)
        if models:
            post_model_selector['values'] = models
            if len(models) > 2:
                post_model_selector.set(models[2])  # 設定預設選項llama3.2:3b
            else:
                post_model_selector.set(models[0])  # 如果模型數量不足，設為第一個
    
    fetch_models_button = tk.Button(ollama_frame, text="刷新模型列表", command=update_post_models, width=15, height=1)
    fetch_models_button.grid(row=1, column=2, padx=15, pady=5, sticky="w")

    # 初始化後處理模型列表
    update_post_models()
    
    # 後處理類型選擇
    tk.Label(ollama_frame, text="選擇後處理類型：").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    selected_question_var = tk.StringVar(value="訂正錯字")
    question_options = ["訂正錯字", "訂正錯字+潤飾", "會議摘要", "會議紀錄", "重點摘要", "一行一句", "中文翻譯", "英文翻譯"]
    question_menu = ttk.Combobox(ollama_frame, textvariable=selected_question_var, values=question_options, state="readonly", width=30)
    question_menu.grid(row=2, column=1, sticky="w", padx=5, pady=5)
    tk.Label(ollama_frame, text="給語言模型的提示詞類型").grid(row=2, column=2, padx=15, sticky="e")
    # center_custom_messagebox("Debug", selected_question_var.get())

    # --- 輸出路徑設定區 ---
    output_frame = tk.LabelFrame(root, text="輸出設定", padx=10, pady=10)
    output_frame.pack(padx=10, pady=10, fill="x")

    tk.Label(output_frame, text="輸出路徑：").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    output_path_var = tk.StringVar(value=OUTPUT_DIRECTORY)
    output_path_entry = tk.Entry(output_frame, textvariable=output_path_var, width=50)
    output_path_entry.grid(row=0, column=1, padx=5, pady=5)
    browse_button = tk.Button(output_frame, text="瀏覽", command=lambda: choose_output_path(output_path_var), width=10, height=1)
    browse_button.grid(row=0, column=2, padx=5, pady=5)

    tk.Label(output_frame, text="輸出檔名：").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    output_filename_var = tk.StringVar(value="transcript.txt")
    output_filename_entry = tk.Entry(output_frame, textvariable=output_filename_var, width=50)
    output_filename_entry.grid(row=1, column=1, padx=5, pady=5)

    # --- 按鈕區域 ---
    button_frame = tk.Frame(root)
    button_frame.pack(padx=480, pady=10, fill="x", expand=True)

    # --- 管理詞庫按鈕 ---
    manage_vocab_button = tk.Button(button_frame, text="管理詞庫", command=manage_vocabulary, width=10, height=1)
    manage_vocab_button.pack(side=tk.LEFT, padx=30)
    
    # --- 儲存訂正結果按鈕 ---
    save_button = tk.Button(button_frame, text="儲存訂正結果", command=lambda: save_correction(text_area_edit), width=10, height=1)
    save_button.pack(side=tk.LEFT, padx=30)
      
    # --- 選擇檔案按鈕 ---
    select_button = tk.Button(button_frame, text="選擇音頻檔案", width=12, height=1, command=lambda: [clear_text(), select_file(
        text_area_raw, text_area_edit, post_model_selector, {
            "task": task_var.get(),
            "language": selected_language_var.get(),
            "temperature": temperature_var.get(),
            "best_of": best_of_var.get(),
            "beam_size": beam_size_var.get()
        }, output_path_var, output_filename_var, ollama_server_var, progress_var, selected_question_var)] if validate_inputs() else None)
    select_button.pack(side=tk.LEFT, padx=30)

    # --- 文字顯示區域 ---
    display_frame = tk.Frame(root)
    display_frame.pack(padx=5, pady=0, fill="x", expand=True)
    
    # 原始轉寫結果
    # text_area_raw = scrolledtext.ScrolledText(display_frame, wrap=tk.WORD, width=60, height=30, state=tk.DISABLED, undo=True)
    text_area_raw = scrolledtext.ScrolledText(display_frame, wrap=tk.WORD, width=60, height=25, undo=True)
    text_area_raw.pack(side=tk.LEFT, padx=5, pady=0, fill=tk.BOTH, expand=True)
    #text_area_raw.insert(tk.END, "這是一個唯讀的文字框。請選取文字，然後右鍵複製。")
    text_area_raw.config(state=tk.DISABLED)  # 插入文本後重新設置為唯讀

    # 編輯後的轉寫結果
    text_area_edit = scrolledtext.ScrolledText(display_frame, wrap=tk.WORD, width=60, height=25, undo=True)
    text_area_edit.pack(side=tk.LEFT, padx=5, pady=0, fill=tk.BOTH, expand=True)
    #text_area_edit.insert(tk.END, "這是一個可以用滑鼠右鍵複製內容的文字框。請選取文字，然後右鍵複製。")

    # 創建右鍵選單（唯讀文字框）
    right_click_menu_raw = tk.Menu(root, tearoff=0)
    right_click_menu_raw.add_command(label="複製(C)", command=lambda: copy_to_clipboard(text_area_raw))
    # 由於 text_area_raw 是唯讀的，剪下、貼上、刪除不適用
    right_click_menu_raw.add_separator()
    right_click_menu_raw.add_command(label="全選(A)", command=lambda: select_all(text_area_raw))

    # 創建右鍵選單（可編輯文字框）
    right_click_menu_edit = tk.Menu(root, tearoff=0)
    right_click_menu_edit.add_command(label="複製(C)", command=lambda: copy_to_clipboard(text_area_edit))
    right_click_menu_edit.add_command(label="剪下(T)", command=lambda: text_area_edit.event_generate("<<Cut>>"))
    right_click_menu_edit.add_command(label="貼上(P)", command=lambda: text_area_edit.event_generate("<<Paste>>"))
    right_click_menu_edit.add_command(label="刪除(D)", command=lambda: delete_selection(text_area_edit))
    right_click_menu_edit.add_separator()
    right_click_menu_edit.add_command(label="全選(A)", command=lambda: select_all(text_area_edit))
    right_click_menu_edit.add_separator()
    right_click_menu_edit.add_command(label="復原(Z)", command=lambda: undo_action(text_area_edit))
    right_click_menu_edit.add_command(label="重做(Y)", command=lambda: redo_action(text_area_edit))


     # 綁定右鍵點擊事件
    text_area_raw.bind("<Button-3>", lambda event: popup(event, text_area_raw, right_click_menu_raw))
    text_area_edit.bind("<Button-3>", lambda event: popup(event, text_area_edit, right_click_menu_edit))

    # 綁定快捷鍵（僅對可編輯文字框有效）
    root.bind_all("<Control-z>", lambda event: undo_action(text_area_edit))
    root.bind_all("<Control-y>", lambda event: redo_action(text_area_edit))
    
    
     # --- 文字顯示區域 Label---
    display_label_frame = tk.Frame(root)
    display_label_frame.pack(padx=5, pady=0, expand=True)
    tk.Label(display_label_frame, text="Whisper轉寫結果").pack(side=tk.LEFT, padx=300)
    tk.Label(display_label_frame, text="LLMs後處理結果").pack(side=tk.RIGHT, padx=300) 
    
    # --- 進度條 ---
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(padx=10, pady=10, fill="x")

    return root



# --- 主程序 ---
if __name__ == "__main__":
    root = create_gui()
    root.mainloop()
