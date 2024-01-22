import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
m_path = "C:/Users/infected4098/Desktop/LYJ/librispeech/train"
train_audio = np.load(os.path.join(m_path, "train_audio_padded.npy"))
train_text = np.load(os.path.join(m_path, "train_text.npy"))
test_audio = np.load(os.path.join(m_path, "test_audio_padded.npy"))
test_text = np.load(os.path.join(m_path, "test_text.npy"))
# Mel spectrogram
# print(train_audio.shape, train_text.shape, test_audio.shape, test_text.shape) =
# (5350, 1368, 80) (5350, 302, 30) (2320, 1368, 80) (2320, 302, 30)





class audio_text_dataset(Dataset):

    def __init__(self, X_data, y_data):

        # self.X_data = X_data #(#data, #timesteps, #coefs) : (5362, 2456, 40)
        # self.y_data = y_data #(#data, #timesteps, #output_class_dim) : (5362, 391, 30)
        self.X_torch = torch.FloatTensor(X_data).to(device) # (#data, 1, #coefs, #timetsteps)
        self.y_torch = torch.FloatTensor(y_data).to(device) # (#data, #timesteps, #output_class_dim)
        self.X_len = self.X_torch.shape[0]
        self.y_len = self.y_torch.shape[0]
        assert self.X_len == self.y_len

    def __getitem__(self, index):
        return self.X_torch[index], self.y_torch[index]

    def __len__(self):
        return self.X_len


batch_size = 8
audio_train = audio_text_dataset(train_audio, train_text)
audio_test = audio_text_dataset(test_audio, test_text)
train_loader = DataLoader(audio_train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(audio_test, batch_size = batch_size, shuffle = True)
inference_loader = DataLoader(audio_test, batch_size = 1, shuffle = True)