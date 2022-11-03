# -*- coding: utf-8 -*-
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import re

# en 其他语料同理
# path1='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\2019-2020\\en_2019_2020.xlsx'
# path2='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\2021\\en_Hasoc2021_train.csv'
# save_path = 'D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\en\\en-2019-2020-2021.csv'
# train_data_path='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\en\\train.csv'
# dev_data_path='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\en\\dev.csv'

# # en
# path1='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2019-2020/en_2019_2020.xlsx'
# path2='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2021/en_Hasoc2021_train.csv'
# save_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/en-2019-2020-2021.csv'
# train_data_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/train.csv'
# dev_data_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/dev.csv'

# #hi 其他语料同理
# path1='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\2019-2020\\hi_2019_2020.xlsx'
# path2='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\2021\\hi_Hasoc2021_train.csv'
# save_path = 'D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\hi\\hi-2019-2020-2021.csv'
# train_data_path='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\hi\\train.csv'
# dev_data_path='D:\\PycharmProjects\\hasoc2021\\Data\\task1_data\\subtask1-B\\sub-B_process\\hi\\dev.csv'

# en
path1 = '/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2019-2020/en_2019_2020.xlsx'
path2 = '/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2021/en_Hasoc2021_train.csv'
save_path = '/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/en-2019-2020-2021.csv'
train_data_path = '/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/train.csv'
dev_data_path = '/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/en/dev.csv'

# hi
# path1='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2019-2020/hi_2019_2020.xlsx'
# path2='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/2021/hi_Hasoc2021_train.csv'
# save_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/hi/hi-2019-2020-2021.csv'
# train_data_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/hi/train.csv'
# dev_data_path='/content/drive/MyDrive/hasoc2021/Data/task1_data/subtask1-B/sub-B_process/hi/dev.csv'


def Rep(text):  #对要提取的数据 依次正则 清理数据
    # HTTP标签
    text = re.sub('\ud83c','',text)
    text = re.sub('\ud83d','',text)
    text = re.sub('\ud83e','',text)
    text = re.sub('0xc5','',text)

    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
    text = pattern.sub('', text)

    # 表情符号
    pattern1 = re.compile("[^(\u2E80-\u9FFF\\w\\s`~!@#\\$%\\^&\\*\\(\\)_+-？（）——=\\[\\]{}\\|;。，、《》”：；“！……:'\"<,>\\.?/\\\\*)]")
    text = pattern1.sub('', text)

    # #标点符号
    # r = "[+-.=?:;—„,$%^，。、~￥%……《》<>「」{}【】()/\\\[\]\"]"
    # text = re.sub(r, ' ', text)


    text=re.sub('☔','',text) #  将#号或【号替换为空“
    text = re.sub('//', '', text) #将@的人名去掉

    #再对特殊的符号进行清洗
    text = re.sub('!','',text)
    text = re.sub('❗', '', text)
    text = re.sub('❌', '', text)
    text = re.sub('➡️', '', text)
    text = re.sub('\n','',text)
    text = re.sub('●','',text)
    text = re.sub('❤','',text)
    text = re.sub('€','',text)
    text = re.sub('_','',text)
    text = re.sub('”','',text)
    text = re.sub('⬇','',text)
    text = re.sub('✊','',text)
    return text  #将干净的数据返回出去

def clearing_data():
    data1=pd.read_excel(path1)
    data2=pd.read_csv(path2)

    data_id=[]
    for x in data1['text_id']:
      data_id.append(x)
    for y in data2['_id']:
      data_id.append(y)

    data_label=[]
    for x in data1['task_2']:
      data_label.append(x)
    for y in data2['task_2']:
      data_label.append(y)

    data_text=[]
    for x in data1['text']:
      data_text.append(Rep(x)) #数据清洗
    for y in data2['text']:
      data_text.append(Rep(y))

    # data1=pd.read_excel(path1)
    # ids1 = data1['text_id'].tolist()
    # label1 = data1['task_1'].tolist()

    # data2=pd.read_csv(path2,encoding='utf-8')
    # ids1 = data2['_id'].tolist()
    # label1 = data2['task_1'].tolist()

    # text1=[]
    # for t in data2['text']:
    #   text1.append(Rep(t))
    pd.DataFrame({'text_id': data_id, 'text': data_text, 'task_2': data_label}).to_csv(save_path, index=False)


def delete_none():
    data=pd.read_csv(save_path)  # 将task2标签为None的行删除
    task2_id=[]
    task2_text=[]
    task2_label=[]

    data = data[data['task_2'] != 'NONE']
    data.to_csv(save_path, index=False)

def split_data():
    #以下为train和dev的划分
    data=pd.read_csv(save_path)

    count=int(len(data)/5) *4 # 取百分之80 20作为训练集与验证集

    from sklearn.utils import shuffle  #打乱数据
    data = shuffle(data)

    train_pd=data[0:count]
    dev_pd=data[count:len(data)]

    # 划分结束后 写回
    train_pd.to_csv(train_data_path, index=False)
    dev_pd.to_csv(dev_data_path, index=False)


if __name__ == '__main__':
    clearing_data()
    # delete_none()
    split_data()


