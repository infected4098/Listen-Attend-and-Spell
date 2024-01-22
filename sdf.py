import torch
import torch.nn as nn
torch.manual_seed(156)
def find_padding_start(label_tensor):
    batch_size, sequence_length, encoding_dim = label_tensor.size()

    # Create a mask of padded elements
    padding_mask = (torch.sum(label_tensor, dim=-1) != 0).float()

    # Find the indices of the first padded element for each sequence
    pad_start_points = (padding_mask).argmax(dim =1)

    return padding_mask
import torchaudio

# Example usage:
# Assuming label_tensor is your onehot encoded label tensor of shape [batch_size, sequence_length, encoding dimension]
# Replace this tensor with your actual data

label_tensor = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 2, 0], [0, 0, 0]],  # Example sequence 1
                             [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]])  # Example sequence 2

# Assuming zeros at the end represent padding
pad_start_points = find_padding_start(label_tensor)


#print(pad_start_points, pad_start_points.sum(dim = 1))


def maskNLLloss(y_pred, y_true, mask):
    feature_size = y_pred.shape[-1]
    #y_pred.shape = [batch_size, sequence_length, feature_size]
    #y_true.shape = [batch_size, sequence_length, feature_size]
    #mask.shape = [batch_size, sequence_length]
    n_Total = mask.sum(dim = 1)
    mask = mask.unsqueeze(2).repeat(1, 1, feature_size) #[batch_size, sequence_length, feature_size]
    y_pred = torch.mul(y_pred, mask)
    y_true = y_true.float()
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-10)) / n_Total
    return loss
import scipy.io.wavfile as wav
import numpy as np

f_path = "C:/Users/infected4098/Desktop/LYJ/librispeech/train/32/21625/32-21625-0000.wav"
a = np.load("C:/Users/infected4098/Desktop/LYJ/librispeech/train/7859/102521/7859-102521-0017norm_mel128.npy")


def letter_error_rate(ref_sentence, hyp_sentence):
    # Converting both sentences to lowercase
    ref_sentence = ref_sentence.lower()
    hyp_sentence = hyp_sentence.lower()

    # Calculating the length of each sentence
    ref_length = len(ref_sentence)
    hyp_length = len(hyp_sentence)

    # Creating a matrix to store distances between substrings
    distances = [[0] * (hyp_length + 1) for _ in range(ref_length + 1)]

    for i in range(ref_length + 1):
        for j in range(hyp_length + 1):
            if i == 0:
                distances[i][j] = j
            elif j == 0:
                distances[i][j] = i
            elif ref_sentence[i - 1] == hyp_sentence[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(distances[i - 1][j], distances[i][j - 1], distances[i - 1][j - 1])

    # Calculating the letter error rate
    ler = distances[ref_length][hyp_length] / ref_length

    return ler


def mask_tensor(tensor):
    return (tensor != 0).astype(int)
embedding = nn.Embedding(3, 512)

s = embedding(torch.Tensor(1).type(torch.IntTensor).unsqueeze(0).repeat(8, 1))


1+2
#print(y_pred[:, 1:2, :])
y_true = label_tensor.float()
mask = find_padding_start(y_true)
new = mask.unsqueeze(2).repeat(1, 1, 3)
#loss, items = maskNLLloss(y_pred, y_true, mask)
a = torch.mul(y_pred, new)
b = maskNLLloss(y_pred, y_true, mask)
c = nn.CrossEntropyLoss()(y_pred, y_true)
#print(b, c)
#print("")