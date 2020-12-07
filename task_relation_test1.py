

import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm

maxlen = 128
batch_size = 64

path = "data/"
config_path = path+'wwm_uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = path+'wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = path+'wwm_uncased_L-24_H-1024_A-16/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(l['h'], l['relation'], l['t'])]
            })
    return D


# 加载数据集
# train_data = load_data(path+'train.csv')
valid_data = load_data(path + 'val.csv')
predicate2id = {"place served by transport hub": 0, "mountain range": 1, "religion": 2, "participating team": 3, "contains administrative territorial entity": 4, "head of government": 5, "country of citizenship": 6, "original network": 7, "heritage designation": 8, "performer": 9, "participant of": 10, "position held": 11, "has part": 12, "location of formation": 13, "located on terrain feature": 14, "architect": 15, "country of origin": 16, "publisher": 17, "director": 18, "father": 19, "developer": 20, "military branch": 21, "mouth of the watercourse": 22, "nominated for": 23, "movement": 24, "successful candidate": 25, "followed by": 26, "manufacturer": 27, "instance of": 28, "after a work by": 29, "member of political party": 30, "licensed to broadcast to": 31, "headquarters location": 32, "sibling": 33, "instrument": 34, "country": 35, "occupation": 36, "residence": 37, "work location": 38, "subsidiary": 39, "participant": 40, "operator": 41, "characters": 42, "occupant": 43, "genre": 44, "operating system": 45, "owned by": 46, "platform": 47, "tributary": 48, "winner": 49, "said to be the same as": 50, "composer": 51, "league": 52, "record label": 53, "distributor": 54, "screenwriter": 55, "sports season of league or competition": 56, "taxon rank": 57, "location": 58, "field of work": 59, "language of work or name": 60, "applies to jurisdiction": 61, "notable work": 62, "located in the administrative territorial entity": 63, "crosses": 64, "original language of film or TV show": 65, "competition class": 66, "part of": 67, "sport": 68, "constellation": 69, "position played on team / speciality": 70, "located in or next to body of water": 71, "voice type": 72, "follows": 73, "spouse": 74, "military rank": 75, "mother": 76, "member of": 77, "child": 78, "main subject": 79}
id2predicate = {}

for i in predicate2id.keys():
    id2predicate[predicate2id[i]] = i

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)

subject_model = Model(bert.model.inputs, subject_preds)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extract_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=1e-5)
train_model.compile(optimizer=optimizer)


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    # tokens = tokenizer.tokenize(text, maxlen=maxlen)
    # mapping = tokenizer.rematch(text, tokens)
    token_ids1, _ = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.5)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((subject[0],subject[1]), predicate1,
                             (_start,_end))
                        )
                        # print(token_ids1,subject[0],subject[1],_start,_end)
                        # print(tokenizer.decode(token_ids1[subject[0]:subject[1] + 1]))
                        break
        return [(tokenizer.decode([101]+token_ids1[s[0]:s[1] + 1]+[102]), id2predicate[p], tokenizer.decode([101]+token_ids1[o[0]:o[1] + 1]+[102]))
                for s, p, o in spoes]
    else:
        return []

def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('train/dev_pred.json', 'w', encoding='utf-8')
    with open("data/decoded1.txt") as f1:
        for d in tqdm(f1):
            R = set([spo for spo in extract_spoes(d.strip())])
            
            
            s = json.dumps({
                'text': d,
                'spo_list_pred': list(R)

            },
                           ensure_ascii=False,
                           indent=4)
            f.write(s + '\n')
    f.close()

train_model.load_weights('data/best_model.weights')
evaluate(valid_data)
