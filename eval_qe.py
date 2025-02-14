import os
import sys
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def read_data(filepath, strip=True):
    data = open(filepath, mode='r', encoding='utf-8', newline='\n').readlines()
    if strip:
        data = [x.rstrip() for x in data]
    return data

def write_list(data, fname):
    with open(fname, 'w', encoding='utf-8') as (f):
        for item in data:
            try:
                f.write('%s\n' % item)
            except:
                item = item.encode('utf-8')
                f.write('%s\n' % item)

def divide_chunks(l, n):
    new_data = []
    for i in range(0, len(l), n):
        new_data.append(l[i:i + n])
    return new_data

model_id = "Unbabel/TowerInstruct-7B-v0.2"
cache_dir = "./hf_cache/"
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
tokenizer.padding_side = "right"

src = read_data("data/wmt23.en-de.en")  # Change this to the path of the source file
mt = read_data("") # Change this to the path of the machine translation file. If only one hypothesis is available for a source sentence, then set num_return_sequences to 1. If more, change accordingly. 
batch_size = 1
num_return_sequences = 1
src_batches = divide_chunks(src, batch_size)
mt_batches = divide_chunks(mt, batch_size*num_return_sequences)

prefix = "English:\n"
suffix = "\nGerman:\n "
id2label = {0: "0", 1: "1"}

label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(model_id, device_map="auto",cache_dir=cache_dir, id2label=id2label,label2id=label2id,num_labels=2) 
model.load_adapter("models/qe_ende_mqm") 

idx = 0
scores = []
for src_batch in src_batches:
    mt_batch = mt_batches[idx]
    src_batch = [x for x in src_batch for _ in range(num_return_sequences)]
    texts = []
    prefix_texts = []
    for entry in range(len(src_batch)):
        texts.append(prefix + src_batch[entry] + suffix + mt_batch[entry] + tokenizer.eos_token)
        prefix_texts.append(prefix + src_batch[entry] + suffix)

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    prefix_inputs = tokenizer(prefix_texts, return_tensors="pt", padding=True).to(model.device)
    logits = model(**inputs).logits
    logits = torch.nn.functional.log_softmax(logits, dim=-1)

    for entry in range(len(src_batch)):
        prefix_pad_tokens = torch.sum(prefix_inputs.input_ids[entry] == tokenizer.pad_token_id)
        curr_pad_tokens = torch.sum(inputs.input_ids[entry] == tokenizer.pad_token_id)

        prefix_len = prefix_inputs.input_ids[entry].shape[0] - prefix_pad_tokens
        curr_logits = logits[entry][prefix_len : inputs.input_ids[entry].shape[0] - curr_pad_tokens]
        #curr_score = torch.mean(curr_logits[:,0] - curr_logits[:,1] - 2*curr_logits[:,2]).item()
        curr_score = torch.mean(curr_logits[:,0]).item()
        scores.append(curr_score)
        print(curr_score)
    idx+=1
    print(idx)

output_file = ""  # Change this to the path of the output file
assert output_file != "", "Please provide a valid output file path"
write_list(scores, output_file)
