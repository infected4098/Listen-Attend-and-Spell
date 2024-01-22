import torch
import torch.nn as nn
import numpy as np; import pandas as pd
import os
from preprocess import m_path
from decoder import decode, decode_from_word
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)
from speller import Speller
from dataloader import inference_loader
save_path = "C:/Users/infected4098/Desktop/LYJ"

speller_path = os.path.join(save_path, 'speller.pt')
speller = torch.load(speller_path)
speller.eval()
rev_char_map = pd.read_csv(os.path.join(m_path,'idx2char.csv'), header =0)



for i, data in enumerate(inference_loader):
    if i > 10:
        break

    inf_inputs, inf_labels = data
    del data
    inf_gt = inf_labels[:, :-1, :]
    inf_labels_ = inf_labels[:, 1:, :]
    del inf_labels
    sentence, ll = decode(speller, inf_inputs, rev_char_map)
    print("Prediction is : ", sentence, "Log-likelihood is: ", ll)
    true_sent = decode_from_word(inf_gt, rev_char_map)
    print("True sentence is: ", true_sent)
