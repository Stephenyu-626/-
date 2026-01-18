import os
import datetime
from Seq2Seq import Encoder, Decoder
import tensorflow as tf
import typing

# 代码11-6 模型的参数设置
# 设置参数
data_path = '../data/ids'  # 文件路径
epoch = 501  # 迭代训练次数
batch_size = 15  # 每批次样本数
embedding_dim = 256  # 词嵌入维度
hidden_dim = 512  # 隐层神经元个数
shuffle_buffer_size = 4  # 清洗数据集时将缓冲的实例数
device = -1  # 使用的设备ID，-1即不使用GPU
checkpoint_path = '../tmp/model'  # 模型参数保存的路径
MAX_LENGTH = 50  # 句子的最大词长
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}# 最大输出句子的长度

# 代码11-4 加载词典、数据
# 加载词典
print(f'[{datetime.datetime.now()}] 加载词典...')
data_path = '../data/ids'
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}
table = tf.lookup.StaticHashTable(  # 初始化后即不可变的通用哈希表。
    initializer=tf.lookup.TextFileInitializer(
        os.path.join(data_path, 'all_dict.txt'),
        tf.string,
        tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER
    ),  # 要使用的表初始化程序。有关支持的键和值类型，请参见HashTable内核。
    default_value=CONST['_UNK'] - len(CONST)  # 表中缺少键时使用的值。
)

# 加载数据
print(f'[{datetime.datetime.now()}] 加载预处理后的数据...')

# 构造序列化的键值对字典
def to_tmp(text):
    '''
    text: 文本
    '''
    tokenized = tf.strings.split(tf.reshape(text, [1]), sep=' ')
    tmp = table.lookup(tokenized.values) + len(CONST)
    return tmp

# 增加开始和结束标记
def add_start_end_tokens(tokens):
    '''
    tokens: 列化的键值对字典
    '''
    tmp = tf.concat([[CONST['_BOS']], tf.cast(tokens, tf.int32), [CONST['_EOS']]], axis=0)
    return tmp

# 获取数据
def get_dataset(src_path: str, table: tf.lookup.StaticHashTable) -> tf.data.Dataset:
    '''
    src_path: 文件路径
    table:初始化后不可变的通用哈希表。
    
    '''
    dataset = tf.data.TextLineDataset(src_path)
    dataset = dataset.map(to_tmp)
    dataset = dataset.map(add_start_end_tokens)
    return dataset

# 获取数据
src_train = get_dataset(os.path.join(data_path, 'source.txt'), table)
tgt_train = get_dataset(os.path.join(data_path, 'target.txt'), table)

# 代码11-5 数据准备
# 把数据和特征构造为tf数据集
train_dataset = tf.data.Dataset.zip((src_train, tgt_train))

# 过滤数据实例数
def filter_instance_by_max_length(src: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
    '''
    src: 特征
    tgt: 标签 
    '''
    return tf.logical_and(tf.size(src) <= MAX_LENGTH, tf.size(tgt) <= MAX_LENGTH)

train_dataset = train_dataset.filter(filter_instance_by_max_length)  # 过滤数据
train_dataset = train_dataset.shuffle(shuffle_buffer_size)  # 打乱数据
train_dataset = train_dataset.padded_batch(  # 将数据长度变为一致，长度不足用_PAD补齐
    batch_size,
    padded_shapes=([MAX_LENGTH + 2], [MAX_LENGTH + 2]),
    padding_values=(CONST['_PAD'], CONST['_PAD']),
    drop_remainder=True,
)
# 提升产生下一个批次数据的效率
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) 

# 代码11-6 模型的参数设置
# 模型参数保存的路径如果不存在则新建
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    
# 代码11-11 构建模型
# 建模
print(f'[{datetime.datetime.now()}] 创建一个seq2seq模型...')
encoder = Encoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
decoder = Decoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)

# 代码11-12 构建优化器
# 设置优化器
print(f'[{datetime.datetime.now()}] 准备优化器...')
optimizer = tf.keras.optimizers.Adam()

# 代码11-13 损失函数
# 设置损失函数
print(f'[{datetime.datetime.now()}] 设置损失函数...')
# 损失值计算方式
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# 损失函数
def loss_function(loss_object, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    '''
    loss_object: 损失值计算方式
    real: 真实值
    pred: 预测值
    '''
    # 计算真实值和预测值的误差
    loss_ = loss_object(real, pred)
    # 返回输出并不相等的值，并用_PAD填充
    mask = tf.math.logical_not(tf.math.equal(real, CONST['_PAD']))
    # 数据格式转换为跟损失值一致
    mask = tf.cast(mask, dtype=loss_.dtype)
    
    return tf.reduce_mean(loss_ * mask)  # 返回平均误差
    

# 代码11-14 保存模型参数    
# 设置模型保存
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# 代码11-15 设置训练步
# 训练
def train_step(src: tf.Tensor, tgt: tf.Tensor):
    '''
    src: 输入的文本
    tgt: 标签
    '''
    # 获取标签维度
    tgt_width, tgt_length = tgt.shape
    loss = 0
    # 创建梯度带，用于反向计算导数
    with tf.GradientTape() as tape:
        # 对输入的文本编码
        enc_output, enc_hidden = encoder(src)
        # 设置解码的神经元数目与编码的神经元数目相等
        dec_hidden = enc_hidden
        # 根据标签对数据解码
        for t in range(tgt_length - 1):
            # 更新维度，新增1维
            dec_input = tf.expand_dims(tgt[:, t], 1)
            # 解码
            predictions, dec_hidden, dec_out = decoder(dec_input, dec_hidden, enc_output)
            # 计算损失值
            loss += loss_function(loss_object, tgt[:, t + 1], predictions)
    # 计算一次训练的平均损失值
    batch_loss = loss / tgt_length
    # 更新预测值
    variables = encoder.trainable_variables + decoder.trainable_variables
    # 反向求导
    gradients = tape.gradient(loss, variables)
    # 利用优化器更新权重
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss  # 返回每次迭代训练的损失值

# 代码11-16 训练并保存模型	
print(f'[{datetime.datetime.now()}] 开始训练模型...')
# 根据设定的训练次数去训练模型
for ep in range(epoch):
    # 设置损失值
    total_loss = 0
    # 将每批次的数据取出，放入模型里
    for batch, (src, tgt) in enumerate(train_dataset):
        # 训练并计算损失值
        batch_loss = train_step(src, tgt)
        total_loss += batch_loss
    if ep % 100 == 0:
        # 每100训练次保存一次模型
        checkpoint_prefix = os.path.join(checkpoint_path, 'ckpt')
        checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'[{datetime.datetime.now()}] 迭代次数: {ep+1} 损失值: {total_loss:.4f}')
    

# 代码11-17 模型预测
# 模型预测
def predict(sentence='你好'):
    # 导入训练参数
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    # 给句子添加开始和结束标记
    sentence = '_BOS' + sentence + '_EOS'
    # 读取字段
    with open(os.path.join(data_path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
        all_dict = f.read().split()
    # 构建: 词-->id的映射字典
    word2id = {j: i+len(CONST) for i, j in enumerate(all_dict)}
    word2id.update(CONST)
    # 构建: id-->词的映射字典
    id2word = dict(zip(word2id.values(), word2id.keys()))
    # 分词时保留_EOS 和 _BOS
    from jieba import lcut, add_word
    for i in ['_EOS', '_BOS']:
        add_word(i)
    # 添加识别不到的词，用_UNK表示
    inputs = [word2id.get(i, CONST['_UNK']) for i in lcut(sentence)]
    # 长度填充
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=MAX_LENGTH, padding='post', value=CONST['_PAD'])
    # 将数据转为tensorflow的数据类型
    inputs = tf.convert_to_tensor(inputs)
    # 空字符串，用于保留预测结果
    result = ''
    
    # 编码
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2id['_BOS']], 0)

    for t in range(MAX_LENGTH):
        # 解码
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 预测出词语对应的id
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 通过字典的映射，用id寻找词，遇到_EOS停止输出
        if id2word.get(predicted_id, '_UNK') == '_EOS':
            break
        # 未预测出来的词用_UNK替代
        result += id2word.get(predicted_id, '_UNK')
        dec_input = tf.expand_dims([predicted_id], 0)

    return result # 返回预测结果

print('预测示例: \n', predict(sentence='你好，在吗'))