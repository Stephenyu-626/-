# 情感分析模型加载和推理
import os
import numpy as np
import pandas as pd
import jieba
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import sys

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_MODEL_PATH, SENTIMENT_DICT_PATH, SENTIMENT_SEQ_LENGTH

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.dicts = None
        self.maxlen = SENTIMENT_SEQ_LENGTH
        self.load_model()
    
    def load_model(self):
        """加载模型和词典"""
        try:
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 加载模型
                if os.path.exists(SENTIMENT_MODEL_PATH):
                    self.model = load_model(SENTIMENT_MODEL_PATH)
                    print(f"情感分析模型加载成功: {SENTIMENT_MODEL_PATH}")
                else:
                    print(f"警告: 情感分析模型文件不存在: {SENTIMENT_MODEL_PATH}")
                    self.model = None
            
            # 加载或创建词典
            if os.path.exists(SENTIMENT_DICT_PATH):
                with open(SENTIMENT_DICT_PATH, 'rb') as f:
                    self.dicts = pickle.load(f)
                print(f"情感分析词典加载成功: {SENTIMENT_DICT_PATH}")
            else:
                print(f"警告: 情感分析词典文件不存在: {SENTIMENT_DICT_PATH}")
                print("将使用简化版情感分析（基于关键词）")
                self.dicts = None
        except Exception as e:
            print(f"加载情感分析模型失败: {e}")
            self.model = None
            self.dicts = None
    
    def preprocess_text(self, text):
        """预处理文本"""
        if not text:
            return None
        
        try:
            # 分词
            words = list(jieba.cut(text))
            
            if self.dicts is not None:
                # 使用训练时的词典
                word_ids = []
                for word in words:
                    if word in self.dicts.index:
                        word_ids.append(self.dicts.loc[word, 'id'])
                
                if not word_ids:
                    return None
                
                # 填充序列
                sent = sequence.pad_sequences([word_ids], maxlen=self.maxlen)
                return sent
            else:
                # 简化版：基于关键词的情感分析
                return None
        except Exception as e:
            print(f"文本预处理错误: {e}")
            return None
    
    def predict_with_keywords(self, text):
        """基于关键词的简化情感分析"""
        positive_words = ['好', '棒', '赞', '喜欢', '满意', '不错', '优秀', '完美', '开心', '高兴', 
                         '爱', '美', '棒极了', '太好了', '推荐', '值得', '满意', '赞', '👍']
        negative_words = ['差', '坏', '烂', '讨厌', '失望', '糟糕', '垃圾', '不好', '伤心', '难过',
                          '差劲', '不行', '不推荐', '后悔', '糟糕', '差评', '👎']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return {"sentiment": "正面", "confidence": 0.6 + min(pos_count * 0.1, 0.3)}
        elif neg_count > pos_count:
            return {"sentiment": "负面", "confidence": 0.6 + min(neg_count * 0.1, 0.3)}
        else:
            return {"sentiment": "中性", "confidence": 0.5}
    
    def predict(self, text):
        """预测情感"""
        if self.model is None:
            # 使用关键词方法
            return self.predict_with_keywords(text)
        
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                return self.predict_with_keywords(text)
            
            # 使用CPU进行预测，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 预测
                y_pred = self.model.predict(x_pad, verbose=0)
                sentiment_score = float(y_pred[0][0])
            
            # 二分类：0为负面，1为正面
            if sentiment_score >= 0.5:
                sentiment = "正面"
                confidence = sentiment_score
            else:
                sentiment = "负面"
                confidence = 1 - sentiment_score
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "score": sentiment_score
            }
        except Exception as e:
            print(f"预测错误: {e}")
            return self.predict_with_keywords(text)

# 全局实例
sentiment_analyzer = SentimentAnalyzer()

