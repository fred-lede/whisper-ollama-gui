import os
import json
import pandas as pd
import configparser

# 定義目錄結構
directories = [
    "vocabularies",
    "examples",
    "output",
    "models"
]

# 創建目錄
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"確保目錄存在：{directory}")

# 初始化config.ini文件
config = configparser.ConfigParser()
config_file = "config.ini"

default_config = {
    'Server': {
        'Address': '127.0.0.1',
        'Port': '11434'
    },
    'Models': {
        'ModelsDirectory': os.path.join(os.getcwd(), 'models')
    },
    'Output': {
        'OutputDirectory': os.path.join(os.getcwd(), 'output')
    },
    'Whisper': {
        'SupportedLanguages': 'English, Chinese, Thai, Japanese, Korean, Vietnamese'
    }
}

if not os.path.exists(config_file):
    # 如果配置檔案不存在，創建一個默認配置檔案
    with open(config_file, 'w') as f:
        config.read_dict(default_config)
        config.write(f)
    print("config.ini 文件已創建。")
else:
    # 如果配置檔案已存在，讀取它
    config.read(config_file)
    print("config.ini 文件已存在，已讀取。")
    
    # 檢查並添加缺失的區段和選項
    for section, options in default_config.items():
        if not config.has_section(section):
            config.add_section(section)
            print(f"添加缺失的區段：[ {section} ]")
        for key, value in options.items():
            if not config.has_option(section, key):
                config.set(section, key, value)
                print(f"添加缺失的選項：{key} = {value} 在區段 [ {section} ]")
    
    # 將更新後的配置寫回文件
    with open(config_file, 'w') as f:
        config.write(f)
    print("config.ini 文件已更新。")

# 配置變數
try:
    OLLAMA_SERVICE_URL = config['Server']['Address']
    OLLAMA_SERVICE_PORT = config['Server']['Port']
    OLLAMA_API_URL = f"http://{OLLAMA_SERVICE_URL}:{OLLAMA_SERVICE_PORT}"
    MODELS_DIRECTORY = config['Models']['ModelsDirectory']
    OUTPUT_DIRECTORY = config['Output']['OutputDirectory']
except KeyError as e:
    print(f"配置文件中缺少必要的區段或選項：{e}")
    exit(1)

# 讀取支持的語言
SUPPORTED_LANGUAGES = config['Whisper']['SupportedLanguages'].split(", ")

# 初始化詞庫文件
vocab_file = os.path.join("vocabularies", "professional_vocab.json")
if not os.path.exists(vocab_file):
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
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(empty_vocab, f, ensure_ascii=False, indent=4)
    print("詞庫文件已創建。")
else:
    print("詞庫文件已存在。")

# 初始化訂正數據文件
corrections_file = os.path.join(OUTPUT_DIRECTORY, "corrections.csv")
if not os.path.exists(corrections_file):
    df = pd.DataFrame(columns=["input", "output"])
    df.to_csv(corrections_file, index=False)
    print("訂正數據文件已創建。")
else:
    print("訂正數據文件已存在。")

print("目錄結構和檔案已初始化。")
