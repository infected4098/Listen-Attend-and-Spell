import torch
import os
import torch.optim as optim
import torch.nn as nn
from util import mask_tensor, maskNLLloss
import numpy as np
from speller_dense import Speller
from tqdm import tqdm
from dataloader_dense import train_loader, test_loader
CONFIG = {
    'lr': 0.01,
    'epochs': 26,
    'weight_decay': 0.001
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
save_path = "C:/Users/infected4098/Desktop/LYJ"
early_stopping_count = 0
early_stop_thrs = 10
speller = Speller(32, "LSTM", 128, 256, 512,
                  6, 384,  302).to(device)
total_count = 0
for name, param in speller.named_parameters():
    # print(f"Layer: {name} - Size: {param.numel()}")
    total_count += param.numel()
print("#Total Parameters of this model is :", total_count)
loss_history = []
val_loss_history = []
attention_history = []
init_loss = 99999

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(speller.parameters(), lr = CONFIG["lr"], weight_decay = CONFIG["weight_decay"])

for epoch in tqdm(range(CONFIG["epochs"])):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        speller.train()
        inputs, labels = data
        del data
        ground_truth = labels[:, :-1]
        labels_ = labels[:, 1:] # Target sequence
        mask = mask_tensor(labels_)
        del labels
        optimizer.zero_grad(set_to_none = True)

        outputs, attention_scores = speller(inputs, ground_truth, teacher_force_rate = 1)
        # attention_history.append(attention_scores)
        del attention_scores
        loss = maskNLLloss(outputs, labels_, mask).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(speller.parameters(), 5 ) #gradient clipping
        optimizer.step()
        run_loss += loss.item()
        del loss, outputs
        if i % 100 == 99:
            loss_history.append(run_loss/100)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 99:.3f}')
            run_loss = 0.0
            speller.eval()
            with torch.no_grad():
                val_loss = 0.0
                for k, val_data in enumerate(test_loader):
                    val_inputs, val_labels = val_data
                    del val_data
                    val_gt = val_labels[:, :-1]
                    val_labels_ = val_labels[:, 1:]
                    del val_labels
                    mask = mask_tensor(val_labels_)
                    val_output, attention_scores = speller(val_inputs, val_gt)
                    v_loss = maskNLLloss(val_output, val_labels_, mask).mean()
                    val_loss += v_loss
                    del v_loss, val_output, attention_scores, mask
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / 99:.3f}')
                val_loss_history.append(val_loss.item() / 100)
            if val_loss < init_loss:
                torch.save(speller, os.path.join(save_path, 'speller.pt'))

                init_loss = val_loss
                early_stopping_count = 0
            else:
                early_stopping_count += 1

            if early_stopping_count >= early_stop_thrs:
                print("Early Stopping, Best epoch is {}".format(epoch))
                break
            del val_loss

    del run_loss
    torch.cuda.empty_cache()

np.save("C:/Users/infected4098/Desktop/LYJ/las_val_loss_history.npy", val_loss_history)
np.save("C:/Users/infected4098/Desktop/LYJ/las_loss_history.npy", loss_history)


