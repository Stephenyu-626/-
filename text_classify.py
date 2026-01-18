# 文本分类模型加载和推理
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# 配置TensorFlow使用CPU，避免GPU / CUDA 相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    # 某些环境下可能没有 GPU，忽略此错误
    pass

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEXT_CLASSIFY_MODEL_PATH,
    TEXT_CLASSIFY_VOCAB_PATH,
    TEXT_CLASSIFY_CATEGORIES,
    TEXT_CLASSIFY_SEQ_LENGTH,
)

# 与训练脚本一致的清洗逻辑，避免推理阶段分布漂移
import re
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    text = re.sub(r'\s+', '', text)
    return text


# ===== 自定义层和指标（需与训练脚本 10_3_1.py 中保持一致，确保反序列化不报错） =====
class AttentionLayer(tf.keras.layers.Layer):
    """与训练脚本中一致的注意力层，用于加载已训练模型"""

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values):
        # values: (batch, time, dim)
        query = tf.reduce_mean(values, axis=1)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(
            tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))
        )
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class F1Score(tf.keras.metrics.Metric):
    """与训练脚本中一致的 F1 指标，实现自定义指标的反序列化"""

    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        unique_classes = tf.unique(tf.concat([y_true, y_pred], axis=0))[0]
        for i in unique_classes:
            tp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)),
                    tf.float32,
                )
            )
            fp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.not_equal(y_true, i), tf.equal(y_pred, i)),
                    tf.float32,
                )
            )
            fn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)),
                    tf.float32,
                )
            )
            self.true_positives.assign_add(tp)
            self.false_positives.assign_add(fp)
            self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        f1 = 2 * precision * recall / (
            precision + recall + tf.keras.backend.epsilon()
        )
        return f1

    def reset_state(self):
        # 与训练脚本保持同名方法，Keras 会在每个 epoch 调用
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

class TextClassifier:
    def __init__(self):
        self.model = None
        self.words = None
        self.word_to_id = None
        self.categories = TEXT_CLASSIFY_CATEGORIES
        self.seq_length = TEXT_CLASSIFY_SEQ_LENGTH
        self.load_model()
    
    def open_file(self, filename, mode='r'):
        """打开文件"""
        return open(filename, mode, encoding='utf-8', errors='ignore')
    
    def read_vocab(self, vocab_dir):
        """读取词汇表"""
        with self.open_file(vocab_dir) as fp:
            words = [i.strip() for i in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id
    
    def load_model(self):
        """加载模型和词汇表"""
        try:
            # 读取词汇表
            if os.path.exists(TEXT_CLASSIFY_VOCAB_PATH):
                self.words, self.word_to_id = self.read_vocab(TEXT_CLASSIFY_VOCAB_PATH)
            else:
                raise FileNotFoundError(f"词汇表文件不存在: {TEXT_CLASSIFY_VOCAB_PATH}")
            
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 优先加载最佳模型，如果不存在则加载最终模型
                best_model_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_model.h5')
                custom_objects = {
                    "AttentionLayer": AttentionLayer,
                    "F1Score": F1Score,
                }
                
                if os.path.exists(best_model_path):
                    self.model = load_model(best_model_path, custom_objects=custom_objects)
                    print(f"文本分类模型加载成功（最佳模型）: {best_model_path}")
                elif os.path.exists(TEXT_CLASSIFY_MODEL_PATH):
                    self.model = load_model(TEXT_CLASSIFY_MODEL_PATH, custom_objects=custom_objects)
                    print(f"文本分类模型加载成功（最终模型）: {TEXT_CLASSIFY_MODEL_PATH}")
                else:
                    # 尝试其他可能的路径
                    alt_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_validation_best.h5')
                    if os.path.exists(alt_path):
                        self.model = load_model(alt_path, custom_objects=custom_objects)
                        print(f"文本分类模型加载成功: {alt_path}")
                    else:
                        raise FileNotFoundError(f"模型文件不存在。尝试过的路径: {best_model_path}, {TEXT_CLASSIFY_MODEL_PATH}")
        except Exception as e:
            print(f"加载文本分类模型失败: {e}")
            self.model = None
    
    def preprocess_text(self, text):
        """预处理文本（与训练侧一致的清洗）"""
        if not text:
            return None

        text = clean_text(text)
        if not text:
            return None

        # 将文本转换为字符列表
        content = list(text)
        # 转换为ID序列
        data_id = [self.word_to_id.get(x, 0) for x in content if x in self.word_to_id]

        if not data_id:
            return None

        # 使用 pad_sequences 填充到固定长度（与训练代码保持一致）
        x_pad = keras.preprocessing.sequence.pad_sequences(
            [data_id],
            maxlen=self.seq_length,
            padding='post',
            truncating='post'
        )
        return x_pad
    
    def predict(self, text, temperature: float = 0.9):
        """预测文本类别，增加温度与低置信度提示，避免全部同一类"""
        if self.model is None:
            return {"error": "模型未加载"}
        
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                return {"error": "文本预处理失败"}
            
            # 使用CPU进行预测，避免GPU相关错误
            with tf.device('/CPU:0'):
                logits = self.model.predict(x_pad, verbose=0)[0]

                # 温度缩放，减小过度自信或一边倒预测（略低温度提高判别力）
                logits = logits / max(min(temperature, 1.5), 0.5)
                exp_logits = np.exp(logits - np.max(logits))
                y_pred = exp_logits / np.sum(exp_logits)

                # 计算熵与前二差距，检测不确定性
                p_sorted = np.sort(y_pred)[::-1]
                p1 = p_sorted[0]
                p2 = p_sorted[1] if len(p_sorted) > 1 else 0.0
                margin = p1 - p2
                entropy = -np.sum(y_pred * np.log(y_pred + 1e-12))

                predicted_class_idx = int(np.argmax(y_pred))
                confidence = float(p1)
                predicted_class = self.categories[predicted_class_idx]

                # 低置信度或差距过小时给出提示，避免“全一样”假象
                note = ""
                if confidence < 0.35 or margin < 0.05 or entropy > 1.8:
                    note = "预测不确定，请提供更多上下文或更长的文本。"

                probabilities = {
                    self.categories[i]: float(y_pred[i]) 
                    for i in range(len(self.categories))
                }
                
                return {
                    "category": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "note": note
                }
        except Exception as e:
            error_msg = str(e)
            # 如果是GPU相关错误，提供更友好的提示
            if "stream" in error_msg.lower() or "gpu" in error_msg.lower():
                return {"error": "模型预测时发生设备错误，请稍后重试"}
            return {"error": f"预测失败: {error_msg}"}

# 全局实例
text_classifier = TextClassifier()

