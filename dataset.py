import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os

######Data Path######
m_path = "C:/Users/infected4098/Desktop/LYJ/librispeech/train"
m_path_test = "C:/Users/infected4098/Desktop/LYJ/librispeech/test/test-clean"
train_path = os.path.join(m_path, "train.csv")
test_path = os.path.join(m_path, "test.csv")
char_map_path = os.path.join(m_path, "idx2char.csv")


######Audio Dataset Building######
def load_dataset(data_path):
    data_table = pd.read_csv(data_path, index_col=0)
    return data_table
def get_data(data_table, i):
    return np.load(data_table.loc[i]["input"]).T
def ZeroPadding_audio(x,pad_len = 1368):
    features = x.shape[1]
    new_x = np.zeros((pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:len(ins)] = ins
    return new_x


######Text Dataset Building######
def get_text(data_table, i):
    input_string = data_table.loc[i]["label"]
    return list(map(int, re.findall(r'\d+', input_string)))

#print(get_text(data_table,0)) #[2, 3, 4, 5, 4, 6, 7, 8, 9, 6, 10, 11, 6, 12 ... ]

def onehotencoding_pad(x_lst, length, cnt):
    zeros = np.zeros([length+2, cnt])
    zeros[0, 0] = 1

    for i in range(len(x_lst)):
        zeros[i+1, x_lst[i]] = 1
    zeros[i+2, 1] = 1

    return zeros


######Train Data building######

idx2char = pd.read_csv(char_map_path)
data_table = load_dataset(train_path)
long_len = len(get_text(data_table, 5)) #300
idxcnt = idx2char.shape[0]
raw_norm = get_data(data_table, 3)

if not os.path.exists(os.path.join(m_path, "train_audio_padded_mel.npy")):
    cor_idx_lst = []
    train_X_array = np.zeros([data_table.shape[0]-5, 1368, 80])
    train_text = np.zeros([data_table.shape[0]-5, long_len+2, idxcnt])
    for i in tqdm(range(5, data_table.shape[0])):
        raw_norm = get_data(data_table, i)
        raw_norm_padded = ZeroPadding_audio(raw_norm)
        text_qt = get_text(data_table, i)

        if len(text_qt) > long_len:
            continue

        else:
            cor_idx_lst.append(i-5)
            train_X_array[i-5, :, :] = raw_norm_padded
            train_text[i-5, :, :] = onehotencoding_pad(text_qt, long_len, idxcnt)

    train_X_array = train_X_array[cor_idx_lst, :, :]
    train_text = train_text[cor_idx_lst, :, :]
    np.save(os.path.join(m_path, "train_audio_padded_mel.npy"), train_X_array) #train_audio_padded.shape = [5350, 1368, 80]
    np.save(os.path.join(m_path, "train_text_mel.npy"), train_text) #train_text.shape = [5350, 302, 30]




######Test Data building######

test_data_table = load_dataset(test_path)

if not os.path.exists(os.path.join(m_path, "test_audio_padded_mel.npy")):
    cor_test_lst = []
    test_X_array = np.zeros([test_data_table.shape[0]-300, 1368, 80])
    test_text = np.zeros([test_data_table.shape[0]-300, long_len +2, idxcnt])

    for i in tqdm(range(300, test_data_table.shape[0])):
        raw_norm = get_data(test_data_table, i)
        raw_norm_padded = ZeroPadding_audio(raw_norm)
        text_qt = get_text(test_data_table, i)
        if len(text_qt) > long_len:
            continue

        else:
            cor_test_lst.append(i-300)
            test_X_array[i-300, :, :] = raw_norm_padded
            test_text[i-300, :, :] = onehotencoding_pad(text_qt, long_len, idxcnt)
    test_X_array = test_X_array[cor_test_lst, :, :]
    test_text = test_text[cor_test_lst, :, :]
    np.save(os.path.join(m_path, "test_audio_padded_mel.npy"), test_X_array)  #test_X_array.shape = [2320, 1368, 40]
    np.save(os.path.join(m_path, "test_text_mel.npy"), test_text) #test_text.shape = [2320, 302, 30]
1+2






