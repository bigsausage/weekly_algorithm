from collections import Counter
import numpy as np
import tensorflow.keras as kr


def save_vocab(train_dir, vocab_dir, vocab_size = 5000):
    wordList = []
    with open(train_dir,'r',encoding='utf-8')as f:
        for line in f:
            query, label = line.strip('\n').split('__label__')
            for word in query.split():
                wordList.append(word)
            # todo 变成字符而不是词语
            # for word in ''.join(query.split()):
            #     wordList.append(word)

    counter = Counter(wordList)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    with open(vocab_dir,'w',encoding='utf-8')as f2:
        f2.write('\n'.join(words) + '\n')

def load_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir,'r',encoding='utf-8') as f:
        words = [line.strip() for line in f]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def load_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def load_data(filename):
    contents =[]
    labels = []
    with open(filename,'r',encoding='utf-8')as f:
        for line in f:
            query, label = line.strip('\n').split('__label__')
            queryList = query.split()

            #  todo 变成字符而不是词语
            # queryList =[char for char in ''.join(query.split())]

            contents.append(queryList)
            labels.append(label)
    return contents , labels


def process_file(filename, word_to_id, cat_to_id, max_length=30):
    """将文件转换为id表示"""
    contents, labels = load_data(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # label_id.append(cat_to_id[labels[i]])
        label_id.append(int(labels[i]))

    # 使用keras提供的pad_sequences来将文本pad为固定长度 这里是在句子前pad 0
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]