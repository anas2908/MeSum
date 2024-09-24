"""
Created on Thu Jan 18 16:20:26 2024

@author: arpan
"""

"""
Created on Thu Dec  7 22:21:10 2023

@author: arpan
"""

import json
from datasets import load_metric,Dataset,DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
import os
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

from tqdm import tqdm

from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration

from copy import deepcopy

import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import json
from transformers import DataCollator,DefaultDataCollator
from modeling_meme_bart_anas_third import BartForConditionalGeneration

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

print("\nDEVICE:\t",device)
model_checkpoint = "facebook/bart-large" 
metric = load_metric("rouge.py")

TEST_SUMMARY_ID = 1

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

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):

    train = transform_single_dialogsumm_file(train)

    validation = transform_single_dialogsumm_file(validation)

    test = transform_test_file(test)

    return DatasetDict({"train":train,"validation":validation,"test":test})

max_input_length = 1024 

num_epochs =  70

batch_size =8 

path = "/home/arpan_2121cs33/anas/memes/" 
bert_model = "BART"
config = "try_2_ALLTxt_inTok43"+str(max_input_length)+"_3e-6_batchS_"+str(batch_size)
filename_model= bert_model+"_ep_"+str(num_epochs)+config

print(filename_model)

MODEL_PATH_CHECKPOINT = path+"Model Path/"+filename_model+"_Loss_Checkpoints.pt"

MODEL_PATH = path+"Model Path/"+filename_model
is_cuda = torch.cuda.is_available()

import pickle

filename_dataset="memes.xlsx"
data = pd.read_excel(path+filename_dataset)

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

from sklearn.model_selection import train_test_split
import random

train_size = 0.8
val_size = 0.1
test_size = 0.1

train_data, val_test_data = train_test_split(dataset_full, train_size=train_size, random_state=43)
val_data, test_data = train_test_split(val_test_data, train_size=val_size/(val_size + test_size), random_state=43)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

raw_datasets = transform_dialogsumm_to_huggingface_dataset(train_data,val_data,test_data)

model = BartForConditionalGeneration.from_pretrained(model_checkpoint)  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

for param in model.parameters():
    param.requires_grad = True

max_target_length = 1024 

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

args = Seq2SeqTrainingArguments(
    "BART_LARGE_"+config+"ROG_ep_"+str(num_epochs),

    evaluation_strategy = "epoch",

    eval_steps = 50, 

    learning_rate=3e-6, 

    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_strategy="epoch",

    greater_is_better=True,
    fp16=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch"

)

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

from transformers import DataCollator,DefaultDataCollator
data_collator = DefaultDataCollator()

def is_valid_format(input_string):
    pattern =re.compile(r'^(\d+\sto\s\d+)(,\s\d+\sto\s\d+)*$')
    if pattern.match(input_string):
        ranges=input_string.split(', ')
        for r in ranges:
            start, end = map(int, r.split(' to '))
            if end < start:
                return False
        return True
    else:
        return False
def count_clips(binary_str):
    clips = 0
    in_clip = False

    for bit in binary_str:
        if bit == '1':
            if not in_clip:
                in_clip = True
                clips += 1
        else:
            in_clip = False

    return clips

import nltk
import numpy as np
import re
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v , 4) for k, v in result.items()}

def Average_1(lst): 
    return sum(lst) / len(lst) 

def compute_metrics_F1overlap(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    f1score = []
    for ind in range(len(decoded_labels)):

        binary_timestamps_gold = '0' * 500
        binary_timestamps_generated = '0' * 500

        try:
            for timestamp in decoded_labels[ind].split(', '):

                start, end = map(int, timestamp.split(' to '))

                binary_timestamps_gold = binary_timestamps_gold[:start] + '1' * (end - start + 1) + binary_timestamps_gold[end+1:]

        except TypeError as exc :
            print(exc)

            f1score.append(0.000001)

        except ValueError as vexc :
            print(vexc)

            f1score.append(0.000001)

        try:
            for timestamp in decoded_preds[ind].split(', '):

                start, end = map(int, timestamp.split(' to '))

                binary_timestamps_generated = binary_timestamps_generated[:start] + '1' * (end - start + 1) + binary_timestamps_generated[end+1:]

        except TypeError as exc :
            print("exception type error: ",exc)

            f1score.append(0.000001)

        except ValueError as vexc :
            print("exception Value error: ",vexc)

            f1score.append(0.000001)

        len_gold=(binary_timestamps_gold.count('1'))
        len_generated=(binary_timestamps_generated.count('1'))

        overlapped=sum(g == '1' and g == c for g, c in zip(binary_timestamps_gold, binary_timestamps_generated))

        precision=overlapped/len_generated if len_generated > 0 else 0.000001
        recall=overlapped/len_gold if len_gold > 0 else 0.000001
        fscore_denominator = precision + recall

        fscore = 2 * precision * recall / fscore_denominator if fscore_denominator > 0 else 0.000001

        f1score.append(fscore)

    result = {"F1Score": Average_1(f1score)}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    print("front end result:", result)
    return {k: v for k, v in result.items()} 

global count 
count = 0

def compute_metrics_Rouge_F1overlap(eval_pred):
    global count
    count = 1 + count

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result_rouge = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result_rouge = {key: value.mid.fmeasure for key, value in result_rouge.items()}
    result_rouge = {k: v for k, v in result_rouge.items()}
    f1score = []

    for ind in range(len(decoded_labels)):

        binary_timestamps_gold = '0' * 500
        binary_timestamps_generated = '0' * 500

        try:
            for timestamp in decoded_labels[ind].split(', '):

                start, end = map(int, timestamp.split(' to '))

                binary_timestamps_gold = binary_timestamps_gold[:start] + '1' * (end - start + 1) + binary_timestamps_gold[end+1:]

            print(decoded_preds[ind])
            with open(MODEL_PATH+"LOG.txt","a") as fl: 
                fl.write(str(decoded_preds[ind])+"\n")

        except TypeError as exc :
            print(exc)

            f1score.append(0.000001)

        except ValueError as vexc :
            print(vexc)

            f1score.append(0.000001)

        try:
            for timestamp in decoded_preds[ind].split(', '):

                start, end = map(int, timestamp.split(' to '))

                binary_timestamps_generated = binary_timestamps_generated[:start] + '1' * (end - start + 1) + binary_timestamps_generated[end+1:]

        except TypeError as exc :
            print("exception type error: ",exc)

            f1score.append(0.000001)

        except ValueError as vexc :
            print("exception Value error: ",vexc)

            f1score.append(0.000001)

        len_gold=(binary_timestamps_gold.count('1'))
        len_generated=(binary_timestamps_generated.count('1'))

        overlapped=sum(g == '1' and g == c for g, c in zip(binary_timestamps_gold, binary_timestamps_generated))

        precision=overlapped/len_generated if len_generated > 0 else 0.000001
        recall=overlapped/len_gold if len_gold > 0 else 0.000001
        fscore_denominator = precision + recall

        fscore = 2 * precision * recall / fscore_denominator if fscore_denominator > 0 else 0.000001

        f1score.append(fscore)

    avg_f1score = Average_1(f1score)

    rouge_score = result_rouge["rouge1"]
    with open(MODEL_PATH+"LOG.txt","a") as fl: 
            fl.write(str(f"Epoch: {count}, F1Score: {avg_f1score}, Rouge Score: {rouge_score}")+"\n")

    print(f"Epoch: {count}, F1Score: {avg_f1score}, Rouge Score: {rouge_score}")

    result = {"F1_R1Score": avg_f1score + rouge_score}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    print("front end result:", result)
    return {k: v for k, v in result.items()} 

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset= tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],

    data_collator=collate_fn,
    tokenizer=tokenizer,

    compute_metrics=compute_metrics
)

trainer.train()

out = trainer.predict(tokenized_datasets["test"],num_beams=5)

predictions, labels ,metric= out
print(metric)

decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

with open(path+"test_output.txt","w") as f: 
    for i in decoded_preds:
        print(i)
        f.write(str(i.replace("\n",""))+"\n")

with open(MODEL_PATH+"_meme.json", "a") as outfile:
    outfile.write('[')
    for index, item in enumerate(decoded_preds):
        dictionary = {
            "Gold_meme_cap": str(decoded_labels[index]),
            "Generated_meme_cap": decoded_preds[index]
        }
        print(dictionary)
        if index > 0:
            outfile.write(',')
        json.dump(dictionary, outfile)
    outfile.write(']')

