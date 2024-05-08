# coding:utf-8
'''
# 1.导入数据集
# 2.序列填充
# 3.模型建立

# 4.编译网络模型
# 5.训练网络模型
# 6.评估网络模型

# 7.结果可视化
# 8.应用长短期记忆神经网络模型
'''

# 导入所需要的模块与包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    # 1.导入IMDb数据集，将训练集和测试集存储在相应变量中
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=4000)
    # 显示训练集的第一个元素的特征值
    print("序列填充前的第一个元素：\n", x_train[0])

    # 2.训练集特征值进行序列填充
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=400, truncating='post')
    # 测试集特征值进行序列填充
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=400, truncating='post')
    # 显示训练集序列填充后第一个元素的特征值
    print("序列填充后的第一个元素：\n", x_train[0])

    # 3.模型建立
    model = tf.keras.models.Sequential()  # 构建空的网络模型
    # 创建嵌入层并添加到网络模型model
    model.add(tf.keras.Input(shape=(400,)))
    model.add(tf.keras.layers.Embedding(output_dim=32, input_dim=4000))
    # 创建Dropout层并添加到网络模型model
    model.add(tf.keras.layers.Dropout(0.3))
    # 创建长短期记忆层并添加到网络模型model
    model.add(tf.keras.layers.LSTM(32))
    # 创建Dropout层并添加到网络模型model
    model.add(tf.keras.layers.Dropout(0.3))
    # 创建隐藏层作为输出层
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()  # 显示模型各层的参数信息

    # 4.编译网络模型
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # 5.训练网络模型
    history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
    # 6.评估网络模型
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)

    # 7.结果可视化
    # 读取history的history属性
    loss = history.history['loss']  # 训练集损失函数值
    acc = history.history['accuracy']  # 训练集准确率
    val_loss = history.history['val_loss']  # 验证集损失函数值
    val_acc = history.history['val_accuracy']  # 验证集准确率
    # 创建画布并设置画布的大小
    plt.figure(figsize=(10, 3))
    # 在子图1中绘制损失函数值的折线图
    plt.subplot(121)
    plt.plot(loss, color='b', label='train')
    plt.plot(val_loss, color='r', label='validate')
    plt.ylabel('loss')
    plt.legend()
    # 在子图2中绘制准确率的折线图
    plt.subplot(122)
    plt.plot(acc, color='b', label='train')
    plt.plot(val_acc, color='r', label='validate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 8.应用长短期记忆神经网络模型
    dict = {0: "正面评论", 1: "负面评论"}
    # 调用函数完成情感分析预测结果并显示
    test_text = "The ultimate story of friendship, of hope, and of life, and overcoming adversity. I understand why so many class this as the best film of all time, it isn't mine, but I get it. If you haven't seen it, or haven't seen it for some time, you need to watch it, it's amazing."
    display_predict(test_text, model, dict)


def display_predict(text, model, dict):  # 情感分析预测函数
    print("评论为：", text)
    # 定义分词器对象
    token = tf.keras.preprocessing.text.Tokenizer(num_words=4000)
    token.fit_on_texts(text)  # 分词
    input_seq = token.texts_to_sequences(text)  # 输出向量序列
    test_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='post', maxlen=400,
                                                             truncating='post')
    # 使用网络模型预测文本评价
    pred = model.predict(test_seq)
    print("预测结果为：", dict[np.argmax(pred)])  # 显示预测结果


if __name__ == '__main__':
    main()
