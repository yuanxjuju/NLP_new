import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

# 读取数据
df = pd.read_csv('/Users/luying/Downloads/untitled folder/labels.csv')
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
df = df.drop(columns=['text_ocr', 'humour', 'sarcasm', 'offensive', 'motivational'])

# 数据清洗
cleaned = df.copy()
cleaned.dropna(inplace=True)

# 检查数据是否仍有空值
print("是否存在空值: ", cleaned.isnull().sum())

# 标准化文本数据
def standardization(data):
    data = data.apply(lambda x: x.lower())  # 转为小写
    data = data.apply(lambda x: re.sub(r'\d+', '', x))  # 去除数字
    data = data.apply(lambda x: re.sub(r'.com', '', x, flags=re.MULTILINE))  # 去除网址
    data = data.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # 去除标点符号
    return data

cleaned['text_corrected'] = standardization(cleaned['text_corrected'])

# 检查数据分布情况
print("情感标签分布：")
print(cleaned['overall_sentiment'].value_counts())

# 使用 TextVectorization 将文本转换为整数序列
vocab_size = 10000
sequence_length = 50
vectorize_layer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)

text_ds = np.asarray(cleaned['text_corrected'])
vectorize_layer.adapt(tf.convert_to_tensor(text_ds))

# 训练集与测试集的划分
target = cleaned['overall_sentiment']
target = pd.get_dummies(target)  # 将目标标签 one-hot 编码

X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    cleaned['text_corrected'], target, test_size=0.2, stratify=target)

# 定义模型
embedding_dim = 16

def text_model():
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorize_layer(text_input)
    x = Embedding(vocab_size, embedding_dim)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    semi_final_layer = Dropout(0.5)(x)

    # 使用 softmax 作为多分类问题的激活函数
    prediction_layer = Dense(5, activation='softmax', name='task_a')
    output = prediction_layer(semi_final_layer)

    model = Model(inputs=text_input, outputs=output)
    return model

# 创建模型
model = text_model()

# 定义学习率调度
def decay(epoch):
    if epoch < 3:
        return 1e-1
    elif epoch >= 3 and epoch < 10:
        return 1e-3
    else:
        return 1e-4

# 定义回调函数
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = checkpoint_dir + "/ckpt_{epoch}.weights.h5"
callbacks = [
    TensorBoard(log_dir='./logs'),
    ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    LearningRateScheduler(decay),
]

# 使用梯度裁剪来防止梯度爆炸
optimizer = Adam(clipvalue=1.0)

# 编译模型
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(x={"text": X_text_train},
                    y=y_text_train,
                    batch_size=256,
                    epochs=25,
                    callbacks=callbacks)

# 评估模型
loss, accuracy = model.evaluate(X_text_test, y_text_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# 保存训练历史
df_history = pd.DataFrame(history.history)
print(df_history)
