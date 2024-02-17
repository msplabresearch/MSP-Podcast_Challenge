# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
from tqdm import tqdm
from time import perf_counter

# PyTorch Modules
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
import csv

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
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

files_test3 = [filename for filename in os.listdir(audio_path) if 'test3' in filename]

dtype = "test3"

total_dataset=dict()
total_dataloader=dict()

cur_wavs = utils.load_audio(audio_path, files_test3)
wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
total_dataset[dtype] = utils.CombinedSet([cur_wav_set, files_test3])
total_dataloader[dtype] = DataLoader(
    total_dataset[dtype], batch_size=1, shuffle=False, 
    pin_memory=True, num_workers=4,
    collate_fn=utils.collate_fn_wav_test3
)

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.load_state_dict(torch.load(MODEL_PATH+"/final_ssl.pt"))
ssl_model.eval(); ssl_model.cuda()
########## Implement pooling method ##########
feat_dim = ssl_model.config.hidden_size

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)
    pool_model.load_state_dict(torch.load(MODEL_PATH+"/final_pool.pt"))
else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)

pool_model.eval();
pool_model.cuda()
concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 3, dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()


lm = utils.LogManager()
for dtype in ["test3"]:
    lm.alloc_stat_type_list([f"{dtype}_aro", f"{dtype}_dom", f"{dtype}_val"])

min_epoch=0
min_loss=1e10

lm.init_stat()

ssl_model.eval()
ser_model.eval() 


INFERENCE_TIME=0
FRAME_SEC = 0
for dtype in ["test3"]:
    total_pred = []
    total_utt = []
    for xy_pair in tqdm(total_dataloader[dtype]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        mask = xy_pair[1]; mask=mask.cuda(non_blocking=True).float()
        utts = xy_pair[2]

        
        FRAME_SEC += (mask.sum()/16000)
        stime = perf_counter()
        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state
            ssl = pool_model(ssl, mask)
            emo_pred = ser_model(ssl)

            total_pred.append(emo_pred)
            total_utt.append(utts)
        etime = perf_counter()
        INFERENCE_TIME += (etime-stime)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)

    data = []
    for pred, utt in zip(total_pred, total_utt):
        print(pred)
        pred_values =  pred.cpu().tolist()
        print(pred_values)
        data.append([utt[0], min(max(1, pred_values[0] * 6 + 1), 7),min(max(1, pred_values[2] * 6 + 1), 7),min(max(1, pred_values[1] * 6 + 1), 7)])

    # print(data)
    # Writing to CSV file
    os.makedirs(MODEL_PATH + '/results', exist_ok=True) 
    csv_filename = MODEL_PATH + '/results/' + dtype + '.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FileName", "EmoAct", "EmoVal", "EmoDom"])
        writer.writerows(data)


lm.print_stat()
print("Duration of whole dev+test set", FRAME_SEC, "sec")
print("Inference time", INFERENCE_TIME, "sec")
print("Inference time per sec", INFERENCE_TIME/FRAME_SEC, "sec")

