import torch
from preprocess import m_path
import os
from util import CreateOnehotVariable, letter_error_rate
import numpy as np; import pandas as pd
from speller import Speller
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)
rev_char_map = pd.read_csv(os.path.join(m_path,'idx2char.csv'), header =0)



def beam_search_decode(model, beam_width, listener_tensor):
    assert listener_tensor.shapep[0] == 1
    """
    :param model: Speller
    :param beam_width: Int  
    :param listener_tensor: [1, 2456, 40]. Need single data as input
    :return: 
    """
    with torch.no_grad():
        model.eval()
        init_word = CreateOnehotVariable(torch.cuda.FloatTensor(np.zeros((1, 1))), 30).to(device)
        init_word[:, :, 0] = 1 #<sos> token 만들기

        output, _ = model.forward_step(listener_tensor)
        #output.shape = [1, max_label_len, output_class_dim]

    pass

def decode(model, listener_tensor, rev_char_map):
    with torch.no_grad():
        ans_word = ""
        log_prob = 0.0
        model.eval()
        word_tensor, _ = model(listener_tensor = listener_tensor, teacher_force_rate = 0) #Inference mode
        word_tensor = word_tensor.squeeze(dim = 0)
        #word_tensor.shape = [390, label_dim]
        values = torch.argmax(word_tensor, dim = -1).tolist() #[390, ]
        prob = torch.max(word_tensor, dim = -1).values.tolist() #[390, ]
        arr = np.array(rev_char_map.iloc[values, 1].values).reshape(-1)
        for i, word in enumerate(arr):
            if word == "<eos>":
                break
            if word == "<sos>":
                continue
            ans_word += word
            log_prob += np.log(prob[i])

    return ans_word, log_prob


def decode_from_word(word_tensor, rev_char_map):
    ans_word = ""
    word_tensor = word_tensor.squeeze(dim=0)
    # word_tensor.shape = [390, label_dim]
    values = torch.argmax(word_tensor, dim=-1).tolist()  # [390, ]
    arr = np.array(rev_char_map.iloc[values, 1].values).reshape(-1)
    for i, word in enumerate(arr):
        if word == "<sos>":
            continue
        if word == "<eos>":
            break
        ans_word += word

    return ans_word




