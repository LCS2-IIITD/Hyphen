# -*- coding: utf-8 -*-
import csv
import pandas as pd
import sys
import re

#en 其他语料同理
path1='/content/drive/MyDrive/hasoc2021/Data/test/en_Hasoc2021_test_task1.csv'
save_path1 = '/content/drive/MyDrive/hasoc2021/Data/test/test_data_process/en_test.csv'

#hi 其他语料同理
path2='/content/drive/MyDrive/hasoc2021/Data/test/hi_Hasoc2021_test_task1.csv'
save_path2 = '/content/drive/MyDrive/hasoc2021/Data/test/test_data_process/hi_test.csv'

#mr
path3='/content/drive/MyDrive/hasoc2021/Data/test/hasoc2021_mr_test-blind-2021.csv'
save_path3 = '/content/drive/MyDrive/hasoc2021/Data/test/test_data_process/mr_test.csv'

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

    #标点符号
    r = "[+-.!=?:;—„,$%^，。、~￥%……《》<>「」{}【】()/\\\[\]\"]"
    text = re.sub(r, ' ', text)


    text=re.sub('☔','',text) #  将#号或【号替换为空“
    text = re.sub('//', '', text) #将@的人名去掉

    #再对特殊的符号进行清洗
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

def en_clearing_data():
    data1=pd.read_csv(path1)

    data_id=[]
    for y in data1['_id']:
      data_id.append(y)

    data_text=[]
    for y in data1['text']:
      data_text.append(Rep(y))

    pd.DataFrame({'text_id': data_id, 'text': data_text}).to_csv(save_path1, index=False)

def hi_clearing_data():
    data2=pd.read_csv(path2)

    data_id=[]
    for y in data2['tweet_id']:
      data_id.append(y)

    data_text=[]
    for y in data2['text']:
      data_text.append(Rep(y))

    pd.DataFrame({'text_id': data_id, 'text': data_text}).to_csv(save_path2, index=False)

def mr_clearing_data():
    data3=pd.read_csv(path3)

    data_id=[]
    for y in data3['text_id']:
      data_id.append(y)

    data_text=[]
    for y in data3['text']:
      data_text.append(Rep(y))

    pd.DataFrame({'text_id': data_id, 'text': data_text}).to_csv(save_path3, index=False)

if __name__ == '__main__':
    en_clearing_data()
    hi_clearing_data()
    mr_clearing_data()



