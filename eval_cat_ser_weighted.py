# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import csv
from time import perf_counter
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from transformers import AutoModel

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--model_path", type=str, default="./model/wavlm-large")
parser.add_argument("--pooling_type", type=str, default="MeanPooling")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument('--store_path')
args = parser.parse_args()

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

import json
from collections import defaultdict
config_path = "config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()

# Total number of samples
total_samples = len(train_df)

# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}

print(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)


# Print or return the tensor
print(class_weights_tensor)

total_dataset=dict()
total_dataloader=dict()
for dtype in ["dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    cur_emo_set = utils.CAT_EmoSet(cur_labs)

    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=1, shuffle=False, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.load_state_dict(torch.load(MODEL_PATH+"/best_ssl.pt"))
ssl_model.eval(); ssl_model.cuda()
########## Implement pooling method ##########
feat_dim = ssl_model.config.hidden_size

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)
    pool_model.load_state_dict(torch.load(MODEL_PATH+"/best_pool.pt"))
else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)

pool_model.eval()
pool_model.cuda()
concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 8, dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/best_ser.pt"))
ser_model.eval(); ser_model.cuda()


lm = utils.LogManager()
for dtype in ["dev"]:
    lm.alloc_stat_type_list([f"{dtype}_loss"])

min_epoch=0
min_loss=1e10

lm.init_stat()

ssl_model.eval()
ser_model.eval() 

if not os.path.exists(MODEL_PATH + '/results'):
    os.mkdir(MODEL_PATH + '/results')
INFERENCE_TIME=0
FRAME_SEC = 0
for dtype in ["dev"]:
    total_pred = [] 
    total_y = []
    total_utt = []
    for xy_pair in tqdm(total_dataloader[dtype]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        fname = xy_pair[3]
        
        FRAME_SEC += (mask.sum()/16000)
        stime = perf_counter()
        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state
            ssl = pool_model(ssl, mask)
            emo_pred = ser_model(ssl)

            total_pred.append(emo_pred)
            total_y.append(y)
            total_utt.append(fname)

        etime = perf_counter()
        INFERENCE_TIME += (etime-stime)

    def label_to_one_hot(label, num_classes=8):
        one_hot = ['0.0'] * num_classes
        one_hot[label.item()] = '1.0'
        return ','.join(one_hot)

    data = []
    for y, pred, utt in zip(total_y, total_pred, total_utt):
        one_hot_label = label_to_one_hot(y.cpu())
        pred_values = ', '.join([f'{val:.4f}' for val in pred.cpu().numpy().flatten()])
        data.append([utt[0], one_hot_label, pred_values])

    # Writing to CSV file
    csv_filename = MODEL_PATH + '/results/' + dtype + '.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Label', 'Prediction'])
        writer.writerows(data)


##################################

    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Function to convert string representation of one-hot vectors to numpy arrays
    def string_to_array(s):
        return np.array([float(i) for i in s.strip('\"').split(',')])

    # Convert the string representations to numpy arrays
    df['Label'] = df['Label'].apply(string_to_array)
    df['Prediction'] = df['Prediction'].apply(string_to_array)

    # Use argmax to determine the class with the highest probability
    y_true = np.argmax(np.stack(df['Label'].values), axis=1)
    y_pred = np.argmax(np.stack(df['Prediction'].values), axis=1)

    # Compute metrics
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # Print results
    print(f"F1-Micro: {f1_micro}")
    print(f"F1-Macro: {f1_macro}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save the results in a text file
    with open(MODEL_PATH + '/results/' + dtype + '.txt', 'w') as f:
        f.write(f"F1-Micro: {f1_micro}\n")
        f.write(f"F1-Macro: {f1_macro}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")






    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(total_pred, total_y, class_weights_tensor)
    # Logging
    lm.add_torch_stat(f"{dtype}_loss", loss)


lm.print_stat()
print("Duration of whole dev+test set", FRAME_SEC, "sec")
print("Inference time", INFERENCE_TIME, "sec")
print("Inference time per sec", INFERENCE_TIME/FRAME_SEC, "sec")

os.makedirs(os.path.dirname(args.store_path), exist_ok=True)
with open(args.store_path, 'w') as f:
    for dtype in ["dev", "test"]:
        loss = str(lm.get_stat(f"{dtype}_loss"))
        f.write(loss+"\n")
