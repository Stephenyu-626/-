import tensorflow as tf
import typing

# 代码11-8 定义Encoder端
# 编码
class Encoder(tf.keras.Model):
    # 设置参数
    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int) -> None:
        '''
        vocab_size: 词库大小
        embedding_dim: 词向量维度
        enc_units: LSTM层的神经元数量
        '''
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        # 词嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM层，GRU是简单的LSTM层
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)
    # 定义神经网络的传输顺序
    def call(self, x: tf.Tensor, **kwargs) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        '''
        x: 输入的文本
        '''
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state  # 输出预测结果和当前状态

# 代码11-9 定义BahdanauAttention		
# 注意力机制
class BahdanauAttention(tf.keras.Model):
    # 设置参数
    def __init__(self, units: int) -> None:
        '''
        units: 神经元数据量
        '''
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # 全连接层
        self.W2 = tf.keras.layers.Dense(units)  # 全连接层
        self.V = tf.keras.layers.Dense(1)  # 输出层
    # 设置注意力的计算方式
    def call(self, query: tf.Tensor, values: tf.Tensor, **kwargs) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        '''
        query: 上一层输出的特征值
        values: 上一层输出的计算结果
        '''
        # 维度增加一维
        hidden_with_time_axis = tf.expand_dims(query, 1)  
        # 构造计算方法
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # 计算权重
        attention_weights = tf.nn.softmax(score, axis=1)
        # 计算输出
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights  # 输出特征向量和权重

# 代码11-10 定义Decoder端		
# 解码
class Decoder(tf.keras.Model):
    # 设置参数
    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int):
        '''
        vocab_size: 词库大小
        embedding_dim: 词向量维度
        dec_units: LSTM层的神经元数量
        '''
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        # 词嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 添加LSTM层
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        # 全连接层
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 添加注意力机制
        self.attention = BahdanauAttention(self.dec_units)
        
    # 设置神经网络传输顺序
    def call(self, x: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor) \
            -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        x: 输入的文本
        hidden: 上一层输出的特征值
        enc_output: 上一层输出的计算结果
        '''
        # 计算注意力机制层的结果
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 次嵌入层
        x = self.embedding(x)
        # 词嵌入结果和注意力机制的结果合并
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 添加注意力机制
        output, state = self.gru(x)

        # 输出结果更新维度
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出层
        x = self.fc(output)

        return x, state, attention_weights  # 输出预测结果，当前状态和权重
