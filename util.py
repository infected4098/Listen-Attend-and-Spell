import torch
from torch.autograd import Variable
import editdistance as ed
import numpy as np
torch.manual_seed(156)
def CreateOnehotVariable(input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data #[batch_size, 1] type : FloatTensor
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor) #[batch_size, #timestep = 1, 1] type: LongTensor
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps,
                                         encoding_dim).zero_().scatter_(-1, input_x, 1)).type(input_type)

    return onehot_x #<sos>니까 shape는 [batch_size, 1, label_dim]이고 0번째 Index가 1임.

# @torch.jit.script #pointwise operation efficiency
def maskNLLloss(y_pred, y_true, mask):
    """
    :param y_pred: [batch_size, sequence_length, feature_size]
    :param y_true: [batch_size, sequence_length, feature_size]
    :param mask: [batch_size, sequence_length]
    :return: loss: int
    """
    feature_size = y_pred.shape[-1]
    n_total = mask.sum(dim = 1)
    mask = mask.unsqueeze(2).repeat(1, 1, feature_size) #[batch_size, sequence_length, feature_size]
    y_pred = torch.mul(y_pred, mask)
    y_true = y_true.float()
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-10)) / n_total
    return loss

def find_padding(label_tensor):
    padding_mask = (torch.sum(label_tensor, dim=-1) != 0).float() # mask -> True, no mask -> False
    return padding_mask


def mask_tensor(tensor):
    return (tensor != 0).int()



def TimeDistributed(input_module, input_x): #input_x.shape = [batch_size, timestep, feature_size]
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1)) #[batch_size*timestep, feature_size]
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size,time_steps,-1)

def LetterErrorRate(pred_y, true_y):
    ed_accumalate = []
    for p, t in zip(pred_y, true_y):
        compressed_t = [w for w in t if (w != 1 and w != 0)]

        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)

        ed_accumalate.append(ed.eval(compressed_p, compressed_t) / len(compressed_t))

    return ed_accumalate

def letter_error_rate(ref_sentence, hyp_sentence): # ChatGPT created.
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
"""
# Example usage:
reference = "hello world"
hypothesis = "helo wrold"
ler_result = letter_error_rate(reference, hypothesis)
print(f"Letter Error Rate: {ler_result:.2f}")
"""

def label_smoothing_loss():
    pass


def make_onehot(tensor):
    # [batch_size, sequence_length]
    zeros = np.zeros([tensor.shape[0], tensor.shape[1], 32])
