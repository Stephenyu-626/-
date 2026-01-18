# 机器翻译模型加载和推理
import os
import re
import numpy as np
import tensorflow as tf
import pickle
import sys

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRANSLATE_CHECKPOINT_DIR, TRANSLATE_DATA_PATH, TRANSLATE_TOKENIZER_PATH, TRANSLATE_CONFIG_PATH

class Translator:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.inp_lang = None
        self.targ_lang = None
        self.max_length_targ = None
        self.max_length_inp = None
        self.units = 1024
        self.embedding_dim = 256
        self.BATCH_SIZE = 1  # 推理时batch size为1
        self.load_model()
    
    def preprocess_sentence(self, w):
        """预处理句子"""
        w = re.sub(r'([?.!,])', r' \1 ', w)
        w = re.sub(r"[' ']+", ' ', w)
        w = '<start> ' + w + ' <end>'
        return w
    
    def load_model(self):
        """加载翻译模型"""
        try:
            # 检查tokenizer文件是否存在
            if os.path.exists(TRANSLATE_TOKENIZER_PATH):
                with open(TRANSLATE_TOKENIZER_PATH, 'rb') as f:
                    tokenizer_data = pickle.load(f)
                    self.inp_lang = tokenizer_data['inp_lang']
                    self.targ_lang = tokenizer_data['targ_lang']
                    self.max_length_targ = tokenizer_data['max_length_targ']
                    self.max_length_inp = tokenizer_data['max_length_inp']
                    self.embedding_dim = tokenizer_data.get('embedding_dim', 256)
                    self.units = tokenizer_data.get('units', 1024)
                print(f"翻译模型tokenizer加载成功: {TRANSLATE_TOKENIZER_PATH}")
            else:
                print(f"警告: 翻译模型tokenizer文件不存在: {TRANSLATE_TOKENIZER_PATH}")
                self.inp_lang = None
                self.targ_lang = None
                return
            
            # 检查checkpoint目录是否存在
            if not os.path.exists(TRANSLATE_CHECKPOINT_DIR):
                print(f"警告: 翻译模型检查点目录不存在: {TRANSLATE_CHECKPOINT_DIR}")
                self.encoder = None
                self.decoder = None
                return
            
            # 定义Encoder和Decoder类（与优化后的训练代码保持一致）
            class Encoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                    super(Encoder, self).__init__()
                    self.batch_sz = batch_sz
                    self.enc_units = enc_units
                    # 尝试加载优化后的双向编码器
                    try:
                        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                        self.bidirectional_gru = tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(self.enc_units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform',
                                               dropout=0.2,
                                               recurrent_dropout=0.2)
                        )
                        self.fc = tf.keras.layers.Dense(self.enc_units, activation='tanh')
                        self.use_bidirectional = True
                    except:
                        # 回退到原始单向编码器
                        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                        self.gru = tf.keras.layers.GRU(self.enc_units,
                                                       return_sequences=True,
                                                       return_state=True,
                                                       recurrent_initializer='glorot_uniform')
                        self.use_bidirectional = False
                        
                def call(self, x, hidden):
                    x = self.embedding(x)
                    if self.use_bidirectional:
                        output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[hidden, hidden])
                        state = self.fc(tf.concat([forward_state, backward_state], axis=-1))
                    else:
                        output, state = self.gru(x, initial_state=hidden)
                    return output, state
                def initialize_hidden_state(self):
                    return tf.zeros((self.batch_sz, self.enc_units))
            
            class BahdanauAttention(tf.keras.layers.Layer):
                def __init__(self, units):
                    super(BahdanauAttention, self).__init__()
                    self.W1 = tf.keras.layers.Dense(units)
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)
                def call(self, query, values):
                    hidden_with_time_axis = tf.expand_dims(query, 1)
                    score = self.V(tf.nn.tanh(
                        self.W1(values) + self.W2(hidden_with_time_axis)))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * values
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights
            
            class Decoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                    super(Decoder, self).__init__()
                    self.batch_sz = batch_sz
                    self.dec_units = dec_units
                    # 尝试加载优化后的多层解码器
                    try:
                        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                        self.gru1 = tf.keras.layers.GRU(self.dec_units,
                                                       return_sequences=True,
                                                       return_state=True,
                                                       recurrent_initializer='glorot_uniform',
                                                       dropout=0.2,
                                                       recurrent_dropout=0.2)
                        self.gru2 = tf.keras.layers.GRU(self.dec_units,
                                                       return_sequences=False,
                                                       return_state=True,
                                                       recurrent_initializer='glorot_uniform',
                                                       dropout=0.2,
                                                       recurrent_dropout=0.2)
                        self.fc1 = tf.keras.layers.Dense(dec_units * 2, activation='relu')
                        self.dropout = tf.keras.layers.Dropout(0.3)
                        self.fc2 = tf.keras.layers.Dense(vocab_size)
                        self.use_multilayer = True
                    except:
                        # 回退到原始单层解码器
                        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                        self.gru = tf.keras.layers.GRU(self.dec_units,
                                                       return_sequences=True,
                                                       return_state=True,
                                                       recurrent_initializer='glorot_uniform')
                        self.fc = tf.keras.layers.Dense(vocab_size)
                        self.use_multilayer = False
                    self.attention = BahdanauAttention(self.dec_units)
                    
                def call(self, x, hidden, enc_output):
                    context_vector, attention_weights = self.attention(hidden, enc_output)
                    x = self.embedding(x)
                    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                    if self.use_multilayer:
                        output1, state1 = self.gru1(x, initial_state=hidden)
                        output2, state2 = self.gru2(output1, initial_state=state1)
                        output = tf.reshape(output2, (-1, output2.shape[-1]))
                        output = self.fc1(output)
                        output = self.dropout(output)
                        x = self.fc2(output)
                        return x, state2, attention_weights
                    else:
                        output, state = self.gru(x)
                        output = tf.reshape(output, (-1, output.shape[2]))
                        x = self.fc(output)
                        return x, state, attention_weights
            
            # 创建模型实例
            vocab_inp_size = len(self.inp_lang.word_index) + 1
            vocab_tar_size = len(self.targ_lang.word_index) + 1
            
            self.encoder = Encoder(vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            self.decoder = Decoder(vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            
            # 加载checkpoint（如果存在且可用）
            checkpoint_prefix = os.path.join(TRANSLATE_CHECKPOINT_DIR, 'ckpt')
            checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)

            latest_ckpt = tf.train.latest_checkpoint(TRANSLATE_CHECKPOINT_DIR)
            if latest_ckpt is None:
                print(f"警告: 找不到可用的翻译模型checkpoint，路径: {TRANSLATE_CHECKPOINT_DIR}")
                # 不抛异常，后续自动回退到简化规则翻译
                self.encoder = None
                self.decoder = None
                return

            try:
                checkpoint.restore(latest_ckpt)
                print(f"翻译模型加载成功，使用checkpoint: {latest_ckpt}")
            except Exception as ckpt_err:
                # 典型错误: "Read less bytes than requested"，通常是checkpoint损坏或版本不兼容
                print(f"警告: 翻译模型checkpoint加载失败，将回退到简化规则翻译。原因: {ckpt_err}")
                self.encoder = None
                self.decoder = None
                return
            
        except Exception as e:
            print(f"加载翻译模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.encoder = None
            self.decoder = None
    
    def evaluate(self, sentence, beam_width=1):
        """评估翻译（内部方法，支持beam search）"""
        if self.encoder is None or self.decoder is None or self.inp_lang is None or self.targ_lang is None:
            return None, None
        
        try:
            sentence = self.preprocess_sentence(sentence)
            inputs = [self.inp_lang.word_index.get(i, 0) for i in sentence.split(' ') if i in self.inp_lang.word_index]
            if not inputs:
                return None, None
            inputs = tf.keras.preprocessing.sequence.pad_sequences(
                [inputs], maxlen=self.max_length_inp, padding='post')
            inputs = tf.convert_to_tensor(inputs)
            
            result = ''
            hidden = [tf.zeros((1, self.units))]
            enc_out, enc_hidden = self.encoder(inputs, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)
            
            if beam_width == 1:
                # 贪心搜索
                for t in range(self.max_length_targ):
                    predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
                    predicted_id = tf.argmax(predictions[0]).numpy()
                    
                    if predicted_id in self.targ_lang.index_word:
                        predicted_word = self.targ_lang.index_word[predicted_id]
                        if predicted_word == '<end>':
                            break
                        result += predicted_word + ' '
                    else:
                        break
                    
                    dec_input = tf.expand_dims([predicted_id], 0)
            else:
                # 简化的beam search
                for t in range(self.max_length_targ):
                    predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
                    top_k = tf.nn.top_k(predictions[0], k=min(beam_width, len(self.targ_lang.word_index)))
                    predicted_id = top_k.indices[0].numpy()
                    
                    if predicted_id in self.targ_lang.index_word:
                        predicted_word = self.targ_lang.index_word[predicted_id]
                        if predicted_word == '<end>':
                            break
                        result += predicted_word + ' '
                    else:
                        break
                    
                    dec_input = tf.expand_dims([predicted_id], 0)
            
            return result.strip(), sentence
        except Exception as e:
            print(f"翻译评估错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def simple_translate(self, text, direction='zh2en'):
        """简化版翻译（基于规则和常见词汇）"""
        common_dict = {
            '你好': 'Hello',
            '谢谢': 'Thank you',
            '再见': 'Goodbye',
            '是的': 'Yes',
            '不是': 'No',
            '对不起': 'Sorry',
            '请': 'Please',
            '我': 'I',
            '你': 'You',
            '他': 'He',
            '她': 'She',
            '我们': 'We',
            '他们': 'They',
            '这个': 'This',
            '那个': 'That',
            '是': 'is',
            '有': 'have',
            '在': 'at',
            '的': 'of',
            '了': '',
            '吗': '?',
            '？': '?',
            '。': '.',
            '，': ',',
            '！': '!'
        }
        
        if direction == 'zh2en':
            result = text
            for zh, en in common_dict.items():
                result = result.replace(zh, en + ' ')
            return result.strip()
        else:
            reverse_dict = {v: k for k, v in common_dict.items()}
            result = text
            for en, zh in reverse_dict.items():
                if en:
                    result = result.replace(en, zh)
            return result
    
    def translate(self, text, direction='zh2en'):
        """翻译文本"""
        if not text:
            return {"error": "文本不能为空"}
        
        # 如果模型未加载，使用简化版翻译
        if self.encoder is None or self.decoder is None:
            return {
                "original": text,
                "translated": self.simple_translate(text, direction),
                "direction": direction,
                "method": "简化规则翻译"
            }
        
        try:
            # 根据方向决定输入输出语言
            if direction == 'zh2en':
                # 中文到英文：使用模型翻译
                result, processed_input = self.evaluate(text)
                if result:
                    # 清理结果，移除<start>和<end>标记
                    result = result.replace('<start>', '').replace('<end>', '').strip()
                    return {
                        "original": text,
                        "translated": result,
                        "direction": direction,
                        "method": "Seq2Seq模型翻译"
                    }
                else:
                    # 如果模型翻译失败，使用简化版
                    return {
                        "original": text,
                        "translated": self.simple_translate(text, direction),
                        "direction": direction,
                        "method": "模型翻译失败，使用简化规则"
                    }
            else:
                # 英文到中文：目前使用简化版（可以扩展支持反向翻译）
                return {
                    "original": text,
                    "translated": self.simple_translate(text, direction),
                    "direction": direction,
                    "method": "简化规则翻译（英文→中文）"
                }
        except Exception as e:
            print(f"翻译错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"翻译失败: {str(e)}",
                "original": text,
                "translated": self.simple_translate(text, direction),
                "direction": direction,
                "method": "错误后使用简化规则"
            }

# 全局实例
translator = Translator()
