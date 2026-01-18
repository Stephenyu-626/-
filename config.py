# 配置文件
import os

# Flask 配置
FLASK_HOST = '0.0.0.0'  # 允许外部访问
FLASK_PORT = 5000
FLASK_DEBUG = True

# 豆包 API 配置
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_API_KEY = "48a29225-a258-471c-97e6-4e1ebef8ae35"
DOUBAO_MODEL = "doubao-seed-1-6-250615"

# 模型路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 根据实际项目结构调整路径
NLP_DEEPLEARN_DIR = os.path.join(os.path.dirname(BASE_DIR), 'nlp_deeplearn')

# 文本分类模型路径
TEXT_CLASSIFY_MODEL_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'my_model.h5')
TEXT_CLASSIFY_VOCAB_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'data', 'cnews.vocab.txt')

# 情感分析模型路径（需要根据实际情况调整）
SENTIMENT_MODEL_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'sentiment_model.h5')
SENTIMENT_DICT_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'sentiment_dict.pkl')

# 机器翻译模型路径
TRANSLATE_CHECKPOINT_DIR = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'training_checkpoints')
TRANSLATE_DATA_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'data', 'en-ch.txt')
TRANSLATE_TOKENIZER_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'translate_tokenizers.pkl')
TRANSLATE_CONFIG_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'translate_config.txt')

# 文本分类类别
TEXT_CLASSIFY_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

# 序列长度配置
TEXT_CLASSIFY_SEQ_LENGTH = 600
SENTIMENT_SEQ_LENGTH = 50

