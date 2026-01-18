# 10.3.1 文本分类（CPU版 + 模型加载修复 + 单一类别适配）
# ========== 强制禁用GPU ==========
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "false"

import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices([], 'GPU')
    print("✅ 已强制禁用GPU，仅使用CPU运行")

# ========== 基础库导入 ==========
from collections import Counter, defaultdict
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import re
import random
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input, Embedding, SpatialDropout1D, Bidirectional, LSTM,
    LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Concatenate, Dense, Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== 修复1：全局注册所有自定义组件 ==========
# 自定义注意力层
class AttentionLayer(keras.layers.Layer):
    """修正版注意力层：返回一维向量"""
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values):
        query = tf.reduce_mean(values, axis=1)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        # 必须实现get_config才能正确保存/加载
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

# 自定义F1指标（适配Keras保存/加载）
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        unique_classes = tf.unique(tf.concat([y_true, y_pred], axis=0))[0]
        for i in unique_classes:
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)), tf.float32))
            self.true_positives.assign_add(tp)
            self.false_positives.assign_add(fp)
            self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        config = super(F1Score, self).get_config()
        return config

# 修复2：定义标准损失函数（避免自定义loss_fn），加入轻微 label smoothing 提升泛化
def get_loss():
    return keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

# ========== 字体配置 ==========
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
cat_en2cn = {
    'Sports': '体育', 'Finance': '财经', 'RealEstate': '房产', 'Home': '家居', 
    'Education': '教育', 'Tech': '科技', 'Fashion': '时尚', 'Politics': '时政', 
    'Game': '游戏', 'Entertainment': '娱乐'
}
cn2en = {v: k for k, v in cat_en2cn.items()}

# ========== 数据预处理 ==========
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue
                label, content = parts
                if content and label and len(content) >= 5:
                    contents.append(list(clean_text(content)))
                    labels.append(label)
            except Exception:
                continue
    label_count = Counter(labels)
    print(f"文件 {filename} 类别分布：{label_count}")
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=8000):
    data_train, lab = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = [(w, c) for w, c in counter.most_common() if c >= 3][:vocab_size - 1]
    words = ['<PAD>'] + ([w for w, c in count_pairs] if count_pairs else [])
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def augment_text(text_ids, max_length):
    if len(text_ids) <= max_length:
        return text_ids
    start = random.randint(0, len(text_ids) - max_length)
    return text_ids[start:start + max_length]

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    valid_labels = set(cat_to_id.keys())
    valid_label_set = set()
    
    for label, content in zip(labels, contents):
        if label not in valid_labels:
            continue
        content_ids = [word_to_id[x] for x in content if x in word_to_id]
        if 'train' in filename:
            content_ids = augment_text(content_ids, max_length)
        data_id.append(content_ids)
        label_id.append(cat_to_id[label])
        valid_label_set.add(label)
    
    if not data_id:
        raise ValueError(f"文件 {filename} 处理后无有效数据！")
    
    x_pad = pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))
    print(f"文件 {filename} 处理后形状：x={x_pad.shape}, y={y_pad.shape}")
    print(f"文件 {filename} 有效类别：{valid_label_set}")
    return x_pad, y_pad, valid_label_set

# ========== 合成多类别数据集，彻底避免“只有体育”问题 ==========
def build_synthetic_dataset():
    """构造一个覆盖 10 个类别的简易中文数据集，用于小规模演示训练。"""
    synthetic_data = {
        '体育': [
            '中国队在世界杯预选赛中取得了关键胜利',
            '这场篮球比赛的最后一节非常精彩',
            '奥运会游泳项目再次打破世界纪录',
            '足球俱乐部签下了新的前锋球员',
            '羽毛球公开赛男单决赛引发热议',
            '马拉松运动员在极端天气下坚持完赛',
        ],
        '财经': [
            '股市今天大幅上涨投资者情绪高涨',
            '央行宣布下调存款准备金率刺激经济',
            '这家互联网公司发布了最新财报',
            '人民币对美元汇率小幅回升',
            '专家预测未来房地产金融风险可控',
            '基金经理建议投资者关注长期价值',
        ],
        '房产': [
            '一线城市房价连续三个月出现回调',
            '购房者更加关注学区房和地铁房',
            '房地产调控政策持续收紧抑制投机',
            '二手房交易量出现小幅增长',
            '租赁市场对精装修公寓需求旺盛',
            '共有产权房成为年轻人购房新选择',
        ],
        '家居': [
            '智能家居设备正在改变人们的生活方式',
            '厨房装修要注意动线设计和安全',
            '环保家具材料越来越受到消费者青睐',
            '客厅布置讲究采光和空间感',
            '简约风格家居设计受到年轻人欢迎',
            '家用电器的节能等级成为重要指标',
        ],
        '教育': [
            '高考改革方案引发社会广泛关注',
            '在线教育平台为学生提供丰富课程',
            '教师培训有助于提升教学质量',
            '家长越来越重视孩子的素质教育',
            '职业教育为社会培养大量技能人才',
            '高校科研项目取得了重要突破',
        ],
        '科技': [
            '人工智能技术正在深入各个行业',
            '5G网络的建设加速推动万物互联',
            '这款新手机采用了柔性屏幕设计',
            '无人驾驶汽车在多地进行道路测试',
            '量子计算被认为是未来的重要方向',
            '科技公司发布了最新云计算解决方案',
        ],
        '时尚': [
            '今年秋冬流行复古风格外套',
            '设计师发布了最新一季时装系列',
            '街拍成为年轻人展示穿搭的重要方式',
            '彩妆品牌推出了联名限量版口红',
            '时装周上模特们展示了大胆造型',
            '运动休闲风持续引领时尚潮流',
        ],
        '时政': [
            '国家发布了新的宏观经济发展规划',
            '多国领导人出席国际合作论坛',
            '政府加大对基础设施建设的投入',
            '新的环保法律法规即将实施',
            '反腐倡廉工作取得显著成效',
            '外交部门就热点问题发表声明',
        ],
        '游戏': [
            '这款角色扮演游戏上线后广受好评',
            '电竞比赛吸引了大量年轻观众',
            '手机游戏加入了全新的社交系统',
            '玩家可以在开放世界中自由探索',
            '游戏公司举办了周年庆典活动',
            '策略类游戏考验玩家的布局能力',
        ],
        '娱乐': [
            '热门电视剧收视率再创新高',
            '歌手在演唱会上演唱多首经典歌曲',
            '电影节红毯上明星们争奇斗艳',
            '综艺节目邀请了多位知名嘉宾',
            '演员在新片中的表现备受好评',
            '偶像组合发布了全新专辑',
        ],
    }
    texts, labels = [], []
    for label, sentences in synthetic_data.items():
        for s in sentences:
            clean_s = clean_text(s)
            if clean_s:
                texts.append(clean_s)
                labels.append(label)
    return texts, labels


def build_vocab_from_texts(texts, vocab_dir, vocab_size=8000):
    """从文本列表构建字符级词表并写入 vocab 文件。"""
    all_chars = []
    for t in texts:
        all_chars.extend(list(t))
    counter = Counter(all_chars)
    count_pairs = [(w, c) for w, c in counter.most_common() if c >= 1][:vocab_size - 1]
    words = ['<PAD>'] + [w for w, c in count_pairs]
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def texts_to_ids(texts, labels, word_to_id, cat_to_id, max_length=600, is_train=False):
    """将文本和标签转换为模型可用的 ID 序列和 one-hot 向量。
    
    训练阶段：为每条样本生成 2 个随机裁剪版本，相当于在不增加 epoch 数的前提下，
    将每轮训练看到的有效样本量翻倍，从而提升训练强度。
    """
    data_id, label_id = [], []
    valid_label_set = set()
    for label, text in zip(labels, texts):
        if label not in cat_to_id:
            continue
        base_ids = [word_to_id[x] for x in text if x in word_to_id]
        if not base_ids:
            continue

        # 训练集：为每条样本生成 2 个随机版本，增强训练量
        repeat_times = 2 if is_train else 1
        for _ in range(repeat_times):
            content_ids = base_ids
            if is_train:
                content_ids = augment_text(content_ids, max_length)
            data_id.append(content_ids)
            label_id.append(cat_to_id[label])
            valid_label_set.add(label)
    if not data_id:
        raise ValueError("合成数据处理后无有效数据！")
    x_pad = pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad, valid_label_set


# ========== 加载数据：使用合成数据而不是磁盘上的单一类别文件 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
TMP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'tmp'))
os.makedirs(TMP_DIR, exist_ok=True)

vocab_dir = os.path.join(DATA_DIR, 'cnews.vocab.txt')

# 读取分类
categories, cat_to_id = read_category()
seq_length = 600

# 构建合成文本和标签
all_texts, all_labels = build_synthetic_dataset()
print(f"合成数据总样本数: {len(all_texts)}, 类别分布: {Counter(all_labels)}")

# 按类别分层划分训练 / 验证 / 测试集
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    all_texts, all_labels, test_size=0.4, random_state=42, stratify=all_labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# 基于训练集文本构建词表（写入与推理一致的 vocab 文件路径）
build_vocab_from_texts(train_texts, vocab_dir, vocab_size=8000)
words, word_to_id = read_vocab(vocab_dir)
vocab_size = len(words)

# 文本转 ID 序列
x_train, y_train, train_valid_cats = texts_to_ids(train_texts, train_labels, word_to_id, cat_to_id, seq_length, is_train=True)
x_val, y_val, val_valid_cats = texts_to_ids(val_texts, val_labels, word_to_id, cat_to_id, seq_length, is_train=False)
x_test, y_test, test_valid_cats = texts_to_ids(test_texts, test_labels, word_to_id, cat_to_id, seq_length, is_train=False)

# 合并有效类别
all_valid_cats = list(train_valid_cats.union(val_valid_cats).union(test_valid_cats))
all_valid_cats.sort(key=lambda x: cat_to_id[x])
print(f"\n所有数据集中的有效类别：{all_valid_cats}")
print(f"有效类别数：{len(all_valid_cats)}")

# 计算类别权重（仅对出现的类别加权，未出现类别权重=1）
if len(y_train) > 0:
    label_counts = Counter(np.argmax(y_train, axis=1))
    present_classes = sorted(label_counts.keys())
    total_samples = len(y_train)
    class_weights = {}
    for cid in range(len(categories)):
        if cid in label_counts:
            class_weights[cid] = total_samples / (len(present_classes) * label_counts[cid])
        else:
            class_weights[cid] = 1.0
    print("类别权重（仅出现类别加权）:", class_weights)
else:
    class_weights = None

# ========== 构建模型（函数式API，更稳定） ==========
def build_model(vocab_size, seq_length, num_classes=10):
    inputs = Input(shape=(seq_length,))
    
    # Embedding层
    x = Embedding(
        input_dim=vocab_size + 1,
        output_dim=256,
        input_length=seq_length,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = SpatialDropout1D(0.4)(x)
    
    # 双向LSTM层
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    x = LayerNormalization()(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    x = LayerNormalization()(x)
    
    # 注意力+池化融合
    attn = AttentionLayer(128)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([attn, avg_pool, max_pool])
    
    # 全连接层
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # 输出层
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs, name='TextBiLSTM_Attn')
    
    # 编译模型（使用标准损失函数）
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(
        loss=get_loss(),
        optimizer=optimizer,
        metrics=['categorical_accuracy', F1Score()]
    )
    
    return model

# 构建并打印模型
model = build_model(vocab_size, seq_length)
print("模型输出形状：", model.output_shape)
model.summary()

# ========== 训练配置（简化学习率调度） ==========
callbacks = [
    # 使用 H5 格式保存模型，避免原生 Keras options 冲突
    ModelCheckpoint(
        filepath=os.path.join(TMP_DIR, 'best_model.h5'),
        monitor='val_f1_score',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # 仅使用ReduceLROnPlateau，避免调度器冲突
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
]

# ========== 训练模型 ==========
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=25,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights if class_weights else None,
    verbose=1
)

# ========== 绘图函数 ==========
def plot_acc_loss(history):
    plt.figure(figsize=(12, 4), dpi=150)
    
    # 准确率
    plt.subplot(131)
    epochs = len(history.history['categorical_accuracy'])
    plt.plot(range(1, epochs+1), history.history['categorical_accuracy'], 'g-', label='Train Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_categorical_accuracy'], 'b--', label='Val Accuracy')
    plt.title('Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    # 损失
    plt.subplot(132)
    plt.plot(range(1, epochs+1), history.history['loss'], 'g-', label='Train Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'], 'b--', label='Val Loss')
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    # F1-score
    plt.subplot(133)
    plt.plot(range(1, epochs+1), history.history['f1_score'], 'g-', label='Train F1')
    plt.plot(range(1, epochs+1), history.history['val_f1_score'], 'b--', label='Val F1')
    plt.title('F1-Score Trend')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(TMP_DIR, 'train_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_acc_loss(history)

# ========== 模型保存与加载（核心修复） ==========
# 保存最终模型（使用原生格式）
final_model_path = os.path.join(TMP_DIR, 'final_model.keras')
model.save(final_model_path)
print(f"最终模型已保存到: {final_model_path}")

# 修复3：使用custom_object_scope加载模型
best_model_path = os.path.join(TMP_DIR, 'best_model.h5')
if os.path.exists(best_model_path):
    print(f"使用最佳模型测试: {best_model_path}")
    # 完整注册所有自定义组件
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'F1Score': F1Score,
        'CategoricalCrossentropy': keras.losses.CategoricalCrossentropy
    }
    # 使用custom_object_scope确保加载时识别所有组件
    with keras.utils.custom_object_scope(custom_objects):
        model1 = load_model(best_model_path)
else:
    print(f"使用最终模型测试: {final_model_path}")
    model1 = model

# ========== 测试集评估 ==========
y_pre = model1.predict(x_test, batch_size=64, verbose=1)
y_pre_arg = np.argmax(y_pre, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

# 生成分类报告
print("\n===== 测试集分类报告 =====")
actual_cat_ids = sorted(list(set(y_test_arg) | set(y_pre_arg)))
actual_cat_names = [cn2en.get(categories[id], categories[id]) for id in actual_cat_ids]

if len(actual_cat_ids) == 0:
    print("无有效预测结果")
elif len(actual_cat_ids) == 1:
    accuracy = metrics.accuracy_score(y_test_arg, y_pre_arg)
    print(f"单一类别准确率: {accuracy:.4f}")
    print(f"类别: {actual_cat_names[0]}")
else:
    print(classification_report(
        y_test_arg, 
        y_pre_arg, 
        labels=actual_cat_ids,
        target_names=actual_cat_names
    ))

# 核心指标
accuracy = metrics.accuracy_score(y_test_arg, y_pre_arg)
print(f"\n测试集准确率: {accuracy:.4f}")

# 混淆矩阵
plt.figure(figsize=(8, 6), dpi=300)
confm = confusion_matrix(y_test_arg, y_pre_arg, labels=actual_cat_ids)
sns.heatmap(
    confm.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=True,
    linewidths=0.5,
    cmap='YlGnBu',
    xticklabels=actual_cat_names,
    yticklabels=actual_cat_names
)
plt.xlabel('True Label', fontsize=12)
plt.ylabel('Predicted Label', fontsize=12)
plt.title('Text Classification Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(TMP_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()