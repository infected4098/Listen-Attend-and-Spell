#Dataset
#Librispeech Clean Speech 100 hours of training data. 6.3GB
#Librispeech Clean Speech Test data. 346MB
import os
import soundfile as sf
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as features
from joblib import Parallel, delayed
from tqdm import tqdm
txt_suffix = ".txt"
flac_suffix = ".flac"
import torchaudio
from python_speech_features import logfbank

m_path = "C:/Users/infected4098/Desktop/LYJ/librispeech/train"
m_path_test = "C:/Users/infected4098/Desktop/LYJ/librispeech/test/test-clean"

txt_suffix = ".txt"
flac_suffix = ".flac"
def bring_filenames(m_path):
    txt_f_dct = {}
    for suf_1 in os.listdir(m_path):
        if not suf_1.endswith((".csv", ".npy")):
            path_1 = os.path.join(m_path, suf_1)
            for suf_2 in os.listdir(path_1):
                path_2 = os.path.join(path_1, suf_2)
                txt_f_name = str(suf_1)+"-"+str(suf_2)+".trans.txt"
                txt_f_name = os.path.join(path_2, txt_f_name)
                with open(txt_f_name, "r") as txt_file:
                    for line in txt_file:
                        audio_f = line[:-1].split(' ')[0] + flac_suffix
                        audio_f = os.path.join(path_2, audio_f)
                        txt_f_dct[audio_f] = ' '.join(line[:-1].split(' ')[1:])

    return txt_f_dct
""" {8975-270782-0087.flac': 'OF PERVERSE SEXUAL ACTIVITY TO BE SURE THE GROWN UPS ARE RIGHT IN LOOKING UPON THESE 
THINGS AS CHILDISH PERFORMANCES AS PLAY FOR THE CHILD IS NOT TO BE JUDGED AS MATURE AND ANSWERABLE EITHER BEFORE THE BAR OF
 CUSTOM OR BEFORE THE LAW', '8975-270782-0088.flac': 'BUT THESE THINGS DO EXIST THEY HAVE THEIR SIGNIFICANCE AS INDICATIONS OF 
 INNATE CHARACTERISTICS AS WELL AS CAUSES AND FURTHERANCES OF LATER DEVELOPMENTS THEY GIVE US AN INSIGHT INTO CHILDHOOD SEX LIFE 
 AND THEREBY INTO THE SEX LIFE OF MAN, ....} """

def flac2wav(f_path):
    y, sr = sf.read(f_path)
    sf.write(f_path[:-5]+".wav", y, sr, format = "WAV")

def wav2logfbank(f_path):
    sr, y = wav.read(f_path) # sr은 16000
    fbank = logfbank(y, sr, winlen = 0.025, nfilt = 26)
    np.save(f_path[:-4]+"fb40.npy", fbank)

def wav2mel(f_path):
    y, sr = torchaudio.load(f_path) # sr은 16000
    mels = torchaudio.transforms.MelSpectrogram(sample_rate = 16000, n_fft = 400, n_mels = 80)
    mel_spectrogram = mels(y).numpy().reshape(80, -1)
    np.save(f_path[:-4]+"mel128.npy", mel_spectrogram)


def norm(f_path, mean, std):
    np.save(f_path[:-5]+"normfb40.npy", (np.load(f_path, allow_pickle = True)-mean)/std)

def norm_mel(f_path, mean, std):
    np.save(f_path[:-10]+"norm_mel128.npy", (np.load(f_path, allow_pickle = True)-mean)/std)


print("######Train Data Processing###### \n")
print("################flac2wav#################")
dct = bring_filenames(m_path)
dct_lst = list(dct.keys())
Parallel(n_jobs = -1 , backend = "threading")(delayed(flac2wav)(f_path) for f_path in tqdm(dct_lst))
print("################wav2logfbank#################")
Parallel(n_jobs = -1, backend = "threading")(delayed(wav2mel)(f_path[:-5]+".wav") for f_path in tqdm(dct_lst))

print("######Test Data Processing###### \n")
print("################flac2wav#################")
dct = bring_filenames(m_path_test)
dct_lst = list(dct.keys())
Parallel(n_jobs = -1, backend = "threading")(delayed(flac2wav)(f_path) for f_path in tqdm(dct_lst))
print("################wav2logfbank#################")
Parallel(n_jobs = -1, backend = "threading")(delayed(wav2mel)(f_path[:-5]+".wav") for f_path in tqdm(dct_lst))


print("################Train Data normalizing#################")
dct = bring_filenames(m_path)
dct_lst = list(dct.keys())
X_train = []
X_train_dummy = []
for dc in tqdm(dct_lst):
    X_train.extend(np.load(dc[:-5] + "mel128.npy").reshape(-1).tolist())
    X_train_dummy.append(np.load(dc[:-5] + "mel128.npy").T)
mean_x = np.mean(np.array(X_train))
std_x = np.std(np.array(X_train))
Parallel(n_jobs=-1, backend="threading")(delayed(norm_mel)(f_path[:-5]+"mel128.npy", mean_x, std_x) for f_path in tqdm(dct_lst))

X_len = [x.shape[0] for x in X_train_dummy]
tr_file_lst = [dct_lst[idx] for idx in reversed(np.argsort(X_len))]
tr_text_lst = [dct[str(path)] for path in tr_file_lst]


char_map = {}
char_map['<sos>'] = 0
char_map['<eos>'] = 1
char_idx = 2
for text in tr_text_lst:
    for char in text:
        if char not in char_map:
            char_map[char] = char_idx
            char_idx +=1

rev_char_map = {v:k for k,v in char_map.items()}


with open(m_path+'/idx2char_label.csv','w') as f:
    f.write('idx,char\n')
    for i in range(len(rev_char_map)):
        f.write(str(i)+','+rev_char_map[i]+'\n')

tmp_lst = []
for text in tr_text_lst:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_lst.append(tmp)
tr_text = tmp_lst
del tmp_lst



print("################Train data Developing#################")

with open(m_path+"/train_mel.csv",'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(tr_text)):
        f.write(str(i)+',')
        f.write(tr_file_lst[i][:-5]+"norm_mel128.npy"+',')
        for char in tr_text[i]:
            f.write(' '+str(char))
        f.write('\n')





print("################Test Data normalizing#################")
dct = bring_filenames(m_path_test)
dct_lst = list(dct.keys())
Parallel(n_jobs=-1, backend="threading")(delayed(norm_mel)(f_path[:-5]+"mel128.npy", mean_x, std_x) for f_path in tqdm(dct_lst))

X_test = []
for dc in tqdm(dct_lst):
    X_test.append(np.load(dc[:-5] + "mel128.npy").T)


X_len = [x.shape[0] for x in X_test]
te_file_lst = [dct_lst[idx] for idx in reversed(np.argsort(X_len))]
te_text_lst = [dct[path] for path in te_file_lst]

print("################Test data Developing#################")

tmp_lst = []
for text in te_text_lst:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_lst.append(tmp)
te_text = tmp_lst
del tmp_lst


with open(m_path+"/test_mel.csv",'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(te_text)):
        f.write(str(i)+',')
        f.write(te_file_lst[i][:-5] + "norm_mel128.npy"+',')
        for char in te_text[i]:
            f.write(' '+str(char))
        f.write('\n')
