from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForTokenClassification, LogitsProcessorList, LogitsProcessor
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
import torch
from datasets import load_dataset
import os

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, DataCollatorForLanguageModeling, TrainingArguments, Trainer #get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

import subprocess as sp
import os
from threading import Thread, Timer
import sched, time
import evaluate
from transformers import EarlyStoppingCallback
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from qe_logits_processor import QELogitsProcessor


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
         for i in range(0, len(l), n):
             yield l[i:i + n]

src = read_data("data/wmt23.en-de.en")

model_id = "Unbabel/TowerInstruct-7B-v0.2"
cache_dir = "./hf_cache/"

tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
tokenizer.padding_side = "left" # Inference mode
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to("cuda:0")

id2label = {0: "0", 1: "1"}
#id2label = {0: "0", 1: "1", 2: "2"}
label2id = {v: k for k, v in id2label.items()}
qe_model = AutoModelForTokenClassification.from_pretrained(model_id,cache_dir=cache_dir, id2label=id2label,label2id=label2id, device_map="auto",  max_memory={0: "0GB", 1: "48GB", 2: "48GB", 3: "48GB"})

qe_model.load_adapter("models/qe_ende_mqm") 


prefix = "<|im_start|>user\nTranslate the sentence from English into German.\nEnglish: "
suffix = "\nGerman:"
pe_suffix = "<|im_end|>\n<|im_start|>assistant\n"


prefix_classifier = "English:\n"
suffix_classifier = "\nGerman:\n"

padding="longest"


hyp = []
src_batches = divide_chunks(src, 1)

cnt=0
num_return_sequences=5
num_beams=5
topk=5
alpha=0.5


with torch.no_grad():
    for src_batch in src_batches:

        text = [prefix + src_entry + suffix + pe_suffix for src_entry in src_batch]
        inputs = tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        text_classifier = [prefix_classifier + src_entry + suffix_classifier for src_entry in src_batch]
        text_classifier = [x for x in text_classifier for _ in range(num_beams*topk)]
        logits_processor = LogitsProcessorList(
            [
            QELogitsProcessor(qe_model=qe_model, tokenizer=tokenizer, prompt=text_classifier, topk=topk, num_beams=num_beams, alpha=alpha),
            ]
        )

        output = model.generate(**inputs, num_beams=5, max_new_tokens=1024, eos_token_id =32005, num_return_sequences=num_return_sequences, logits_processor=logits_processor)



        for idx in range(len(src_batch)):
            for entry in range(num_return_sequences):
                curr_hyp = tokenizer.decode(output[idx*num_return_sequences + entry][len(inputs.input_ids[idx]):], skip_special_tokens=True)
                curr_hyp = curr_hyp.split("English:")[0].rstrip()
                hyp.append(curr_hyp.replace("\n", " "))
                cnt+=1
                print(curr_hyp)
                print(cnt)
                

output_file = ""  # Change this to the path of the output file
assert output_file != "", "Please provide a valid output file path"
write_list(hyp, output_file)
