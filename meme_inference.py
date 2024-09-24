"""
Created on Sun Aug 27 17:55:55 2023

@author: arpan
"""

"""
Created on Thu Aug 24 16:41:13 2023

@author: arpan
"""

"""
Created on Tue Jun  6 14:29:42 2023

@author: arpan
"""

import json
from datasets import load_metric,Dataset,DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
import os
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm

from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model
import pandas as pd
import os
from tqdm import tqdm

from copy import deepcopy
import sys

from check_t5 import T5ForConditionalGeneration
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import json
from modeling_meme_bart_anas_third import BartForConditionalGeneration

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

print("\nDEVICE:\t",device)

data = pd.read_excel("/home/arpan_2121cs33/anas/memes/memes.xlsx")

dataset_full=[]

meme_id = data["meme_id"].values
image_caption = data["image_caption"].values
meme_caption = data["meme_caption"].values
ocr = data["ocr"].values
meme_location = data["meme_location"].values

for i in range(len(meme_caption)):

    dataset_full.append({"meme_id": str(meme_id[i]),"image_caption": str(image_caption[i]),
                        "meme_caption": str(meme_caption[i]),"ocr": str(ocr[i]),
                        "meme_location": str(meme_location[i])}) 

def transform_single_dialogsumm_file(file):

    result = {"meme_id":[],"image_caption":[],"meme_caption":[],"ocr":[],"meme_location":[]}  

    for i in range(len(file)):

        result["meme_id"].append(file[i]["meme_id"])
        result["image_caption"].append(file[i]["image_caption"])
        result["meme_caption"].append(file[i]["meme_caption"])
        result["ocr"].append(file[i]["ocr"])
        result["meme_location"].append(file[i]["meme_location"])

    return Dataset.from_dict(result)

def transform_test_file(file):

    result = {"meme_id":[],"image_caption":[],"meme_caption":[],"ocr":[],"meme_location":[]}  

    for i in range(len(file)):

        result["meme_id"].append(file[i]["meme_id"])
        result["image_caption"].append(file[i]["image_caption"])
        result["meme_caption"].append(file[i]["meme_caption"])
        result["ocr"].append(file[i]["ocr"])
        result["meme_location"].append(file[i]["meme_location"])

    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(test):

    test = transform_test_file(test)

    return DatasetDict({"test":test})

model_name = "BART_LARGE_try_2_ALLTxt_inTok431024_3e-6_batchS_8ROG_ep_70" 
checkpoint = "/checkpoint-134950"

model_checkpoint = "/home/arpan_2121cs33/anas/memes/"+str(model_name)+checkpoint

metric = load_metric("rouge.py")

TEST_SUMMARY_ID = 1

def collate_fn(batch):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    mask = [item['attention_mask'] for item in batch]
    video_embeds = [item['video_embedd'] for item in batch] 
    audio_embeds = [item['audio_embedd'] for item in batch] 

    inputs_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in inputs], batch_first=True, padding_value=0)

    labels_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels], batch_first=True, padding_value=0)

    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in mask], batch_first=True, padding_value=0)

    video_embeds_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in video_embeds], batch_first=True, padding_value=0)

    audio_embeds_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in audio_embeds], batch_first=True, padding_value=0)

    return {'input_ids': inputs_text, 'labels': labels_text, 'video_embedd': video_embeds_padded, 'audio_embedd': audio_embeds_padded, 'attention_mask': attention_mask}

max_input_length = 1024

filename_model = model_name

print(filename_model)

MODEL_PATH = "/home/arpan_2121cs33/anas/memes/Model Path/"
is_cuda = torch.cuda.is_available()

import pickle

filename_dataset="memes.xlsx"

model = BartForConditionalGeneration.from_pretrained(model_checkpoint)  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_target_length = 512 * 2

path = "/home/arpan_2121cs33/anas/memes"

from sklearn.model_selection import train_test_split
import random

train_size = 0.8
val_size = 0.1
test_size = 0.1

train_data, val_test_data = train_test_split(dataset_full, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(val_test_data, train_size=val_size/(val_size + test_size), random_state=42)

raw_datasets = transform_dialogsumm_to_huggingface_dataset(test_data)

def preprocess_function(examples):

    inputs =[]
    compressed_vid = []
    compressed_aud=[]
    model_inputs={}
    input_embedd_feat = []

    for (meme_id, image_caption, ocr) in zip(examples["meme_id"],examples["image_caption"],examples["ocr"]):
        inputs.append(str("bot is provided with meme's image caption:"+str(image_caption)+", bot is also provided with meme's ocr: "+str(ocr)+ " bot is also provided with meme visual embeddings and meme audio features if available, bot task is to give the detail description of the meme"))        
        try:

            npy_fileload = torch.from_numpy(np.load('/home/arpan_2121cs33/anas/memes/meme_emb/clip_/'+str(meme_id)+"_clip_features.npy")).float()
            print("Shape of loaded tensor:", npy_fileload.shape)
            if npy_fileload.shape != [1, 512]:

                compressed_vid.append(torch.zeros(1,512))
            else:  
                compressed_vid.append(npy_fileload)

        except Exception as e:
            print(e) 
            compressed_vid.append(torch.zeros(1,512)) 

        try:

            npy_fileload = torch.from_numpy(np.load('/home/arpan_2121cs33/anas/memes/meme_emb/mfccs/'+str(meme_id)+".npy")).float()
            print("Shape of loaded tensor:", npy_fileload.shape)
            if npy_fileload.shape != [1, 216]:

                compressed_aud.append(torch.zeros(1,216))
            else:  
                compressed_aud.append(npy_fileload)

        except Exception as e:
            print(e) 
            compressed_aud.append(torch.zeros(1,216))

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    model_inputs["video_embedd"] = compressed_vid
    model_inputs["audio_embedd"] = compressed_aud
    print("1")
    label_input =[]
    for cap in examples["meme_caption"]: 
        label_input.append(cap)

    print("2")
    with tokenizer.as_target_tokenizer():
        print("3")

        labels = tokenizer(label_input, max_length=max_target_length, truncation=True)

        print("24")
    model_inputs["labels"] = labels["input_ids"]

    print("5")
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

batch_size = 1
args = Seq2SeqTrainingArguments(
    "ROG_",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    save_strategy="epoch",
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch",report_to="wandb"
)
trainer = Seq2SeqTrainer(
    model,
    args,

    data_collator=collate_fn,
    tokenizer=tokenizer,

)

import nltk
import numpy as np

out = trainer.predict(tokenized_datasets["test"],num_beams=5)

predictions, labels ,metric= out
print(metric)

decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]

with open(MODEL_PATH+"memes_43.json", "a") as outfile: 
    outfile.write('[')
    for index, item in enumerate(decoded_preds):
        dictionary = {

            "Generated_meme_cap": decoded_preds[index]
        }
        print(dictionary)
        if index > 0:
            outfile.write(',')
        json.dump(dictionary, outfile)
    outfile.write(']')