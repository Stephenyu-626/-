# 10.3.2 情感分析（CPU版 | 禁用GPU避CuDNN兼容问题）
# 核心修改：强制使用CPU训练，彻底解决CuDNN版本不兼容报错
import os
# ========== 方案一核心配置：禁用GPU ==========
# 强制TensorFlow使用CPU，避开CuDNN版本不兼容问题
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 可选：关闭TensorFlow冗余日志，只显示错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ============================================

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pandas as pd
from tensorflow.keras.preprocessing import sequence
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from tensorflow.keras import Input
import tensorflow as tf
import time
from sklearn import metrics
import pickle
import io

# ===================== 基础配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
TMP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'tmp'))

SENTIMENT_MODEL_PATH = os.path.join(TMP_DIR, 'sentiment_model.h5')
SENTIMENT_DICT_PATH = os.path.join(TMP_DIR, 'sentiment_dict.pkl')
os.makedirs(os.path.dirname(SENTIMENT_MODEL_PATH), exist_ok=True)

# ===================== 核心函数：强制文本读取（忽略后缀） =====================
def load_excel(path):
    """
    彻底修复版：忽略文件后缀，强制按文本文件读取（解决 File is not a zip file）
    核心：不管后缀是 .xls/.xlsx/.csv，都按文本处理，过滤 NUL 字符 + 多编码容错
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在，请检查路径: {path}")

    file_name = os.path.basename(path)
    print(f"正在读取文件: {file_name} (强制文本模式)")

    # 尝试的编码列表（覆盖中文常见编码）
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-16']

    for encoding in encodings:
        try:
            # 1. 二进制读取，过滤 NUL 空字符
            with open(path, 'rb') as f:
                raw_data = f.read()
                cleaned_data = raw_data.replace(b'\x00', b'')
                if not cleaned_data:
                    continue

            # 2. 解码为文本
            text = cleaned_data.decode(encoding, errors='ignore').strip()
            if not text:
                continue

            # 3. 用 StringIO 读取为 DataFrame（自动检测分隔符）
            df = pd.read_csv(
                io.StringIO(text),
                header=None,
                sep=None,
                engine='python',
                on_bad_lines='skip',
                encoding_errors='ignore'
            )
            print(f"✅ 成功读取 {file_name} (编码: {encoding})，数据行数: {len(df)}")
            return df

        except Exception as e:
            print(f"  - 编码 {encoding} 失败: {str(e)[:50]}")
            continue

    # 极端回退：latin-1 解码后再尝试
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        text = raw.decode('latin-1', errors='ignore')
        return pd.read_csv(io.StringIO(text), header=None, sep=None,
                          engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"读取文件 {file_name} 最终失败: {str(e)}")
        raise e

# ===================== 1. 数据读取与预处理 =====================
print("===== 开始读取正负情感语料 =====")
neg = load_excel(os.path.join(DATA_DIR, 'neg.xls'))
pos = load_excel(os.path.join(DATA_DIR, 'pos.xls'))

# 贴标签
neg['mark'] = 0
pos['mark'] = 1
pn_all = pd.concat([pos, neg], ignore_index=True)
print(f"✅ 正负语料合并完成，总数据量: {len(pn_all)} 条")

# ===================== 2. 分词处理（增强容错） =====================
def cut_words(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return []
    return list(jieba.cut(text.strip()))

pn_all[0] = pn_all[0].astype(str)
pn_all['words'] = pn_all[0].apply(cut_words)

print("\n===== 开始读取评论语料 =====")
try:
    comment = load_excel(os.path.join(DATA_DIR, 'sum.xls'))
    comment_col = 'rateContent' if 'rateContent' in comment.columns else comment.columns[0]
    comment = comment[comment_col].notnull()
    comment = comment[comment_col].astype(str)
    comment['words'] = comment[comment_col].apply(cut_words)
    pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)
    print(f"✅ 评论语料读取完成，有效评论数: {len(comment)} 条")
except Exception as e:
    print(f"⚠️ 读取 sum.xls 出错: {str(e)}，仅使用正负情感语料构建词典")
    pn_comment = pn_all['words'].copy()

# ===================== 3. 词语向量化 =====================
print("\n===== 构建词语词典 =====")
all_words = []
for word_list in pn_comment:
    if isinstance(word_list, list):
        all_words.extend(word_list)

word_counts = pd.Series(all_words).value_counts()
word_counts = word_counts[word_counts >= 2]  # 过滤低频词

dicts = pd.DataFrame({
    'count': word_counts,
    'id': range(1, len(word_counts) + 1)
})
print(f"✅ 词典构建完成，总词数: {len(dicts)}")

del all_words, word_counts, pn_comment

def text_to_id(word_list):
    if not isinstance(word_list, list):
        return []
    return [dicts['id'][word] for word in word_list if word in dicts.index]

pn_all['sent'] = pn_all['words'].apply(text_to_id)

# ===================== 4. 数据标准化 =====================
maxlen = 50
print(f"\n===== 标准化数据（最大长度: {maxlen}） =====")
pn_all['sent'] = list(sequence.pad_sequences(
    pn_all['sent'],
    maxlen=maxlen,
    padding='post',
    truncating='post'
))

x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))
x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all,
    test_size=0.25,
    random_state=42,
    stratify=y_all
)
print(f"✅ 数据划分完成：训练集 {x_train.shape} | 测试集 {x_test.shape}")

# ===================== 5. LSTM模型构建（cuDNN 友好） =====================
print("\n===== 构建 LSTM 模型 =====")
model = Sequential(name="Sentiment_LSTM")
model.add(Input(shape=(maxlen,), name="input_layer"))

model.add(Embedding(
    input_dim=len(dicts) + 1,
    output_dim=128,
    mask_zero=True,
    name="embedding_layer"
))
model.add(Dropout(0.2, name="dropout_1"))

model.add(tf.keras.layers.Bidirectional(
    LSTM(
        64,
        dropout=0.2,           # 仅前向 dropout
        recurrent_dropout=0,   # 设为 0 以启用 cuDNN（CPU模式下无影响）
        activation='tanh',
        recurrent_activation='sigmoid',
        dtype='float32'
    ),
    name="bidirectional_lstm"
))

model.add(Dense(128, activation='relu', name="dense_1"))
model.add(Dropout(0.3, name="dropout_2"))
model.add(Dense(64, activation='relu', name="dense_2"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(1, name="output_layer"))
model.add(Activation('sigmoid', name="sigmoid_activation"))

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

# ===================== 6. 模型训练 =====================
print("\n===== 开始训练模型 =====")
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint(SENTIMENT_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

x_train_fit, x_val_fit, y_train_fit, y_val_fit = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

start_time = time.time()
history = model.fit(
    x_train_fit, y_train_fit,
    batch_size=64,
    epochs=10,
    validation_data=(x_val_fit, y_val_fit),
    callbacks=callbacks,
    verbose=1
)
train_time = int(time.time() - start_time)
print(f"✅ 模型训练完成，总耗时: {train_time} 秒")

# ===================== 7. 模型评估 =====================
print("\n===== 模型评估 =====")
y_pred = model.predict(x_test, verbose=0).round().astype(int)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
print("\n分类报告:")
print(metrics.classification_report(y_test, y_pred))
print("\n混淆矩阵:")
print(metrics.confusion_matrix(y_test, y_pred))

# ===================== 8. 保存模型和词典 =====================
final_model_path = SENTIMENT_MODEL_PATH.replace('.h5', '_final.h5')
model.save(final_model_path)
print(f"\n✅ 最终模型已保存至: {final_model_path}")

with open(SENTIMENT_DICT_PATH, 'wb') as f:
    pickle.dump(dicts, f)
print(f"✅ 词语词典已保存至: {SENTIMENT_DICT_PATH}")

print("\n===== 情感分析模型训练全部完成 =====")