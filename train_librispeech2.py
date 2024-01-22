import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from util import CreateOnehotVariable, find_padding, maskNLLloss
import numpy as np; import pandas as pd

from listener import Listener
from attention import MultiHeadAttention
from speller2 import Speller2
from tqdm import tqdm
from dataloader import train_loader, test_loader
CONFIG = {
    'lr': 0.01,
    'epochs': 26,
    'weight_decay': 0.001
}
early_stopping_count = 0
early_stop_thrs = 10
save_path = "C:/Users/infected4098/Desktop/LYJ"

speller2 = Speller2(30, "LSTM", 512, 256, 6, 384,  302).to(device)
total_count = 0
for name, param in speller2.named_parameters():
    #print(f"Layer: {name} - Size: {param.numel()}")
    total_count += param.numel()
print("#Total Parameters of this model is :", total_count)
loss_history = []
val_loss_history = []
attention_history = []
init_loss = 99999

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(speller2.parameters(), lr = CONFIG["lr"], weight_decay = CONFIG["weight_decay"])

for epoch in tqdm(range(CONFIG["epochs"])):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        speller2.train()
        inputs, labels = data
        del data
        ground_truth = labels[:, :-1, :]
        labels_ = labels[:, 1:, :] #Target sequence
        mask = find_padding(labels_)
        del labels
        optimizer.zero_grad(set_to_none = True)

        outputs, attention_scores = speller2(inputs, ground_truth, teacher_force_rate = 1)
        #attention_history.append(attention_scores)
        del attention_scores
        loss = maskNLLloss(outputs, labels_, mask).mean()
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        del loss, outputs
        if i % 100 == 99:
            loss_history.append(run_loss/100)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 99:.3f}')
            run_loss = 0.0
            speller2.eval()
            with torch.no_grad():
                val_loss = 0.0
                for k, val_data in enumerate(test_loader):
                    val_inputs, val_labels = val_data
                    del val_data
                    val_gt = val_labels[:, :-1, :]
                    val_labels_ = val_labels[:, 1:, :]
                    del val_labels
                    mask = find_padding(val_labels_)
                    val_output, attention_scores = speller2(val_inputs, val_gt)
                    v_loss = maskNLLloss(val_output, val_labels_, mask).mean()
                    val_loss += v_loss
                    del v_loss, val_output, attention_scores, mask
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / 99:.3f}')
                val_loss_history.append(val_loss.item() / 100)
            if val_loss < init_loss:
                torch.save(speller2, os.path.join(save_path, 'speller2.pt'))

                init_loss = val_loss
                early_stopping_count = 0

            else:
                early_stopping_count += 1

            if early_stopping_count >= early_stop_thrs:
                print("Early Stopping, Best epoch is {} and the best validation loss is {}".format(epoch, val_loss))
                break
            del val_loss
    del run_loss
    torch.cuda.empty_cache()

np.save("C:/Users/infected4098/Desktop/LYJ/2_las_val_loss_history.npy", val_loss_history)
np.save("C:/Users/infected4098/Desktop/LYJ/2_las_loss_history.npy", loss_history)


