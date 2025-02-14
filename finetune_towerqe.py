from transformers import AutoModelForTokenClassification, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from peft import prepare_model_for_kbit_training
import torch
from datasets import load_dataset
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_PROJECT'] = 'towerqe'
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, DataCollatorForTokenClassification, TrainingArguments, Trainer #get_linear_schedule_with_warmup
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



model_id = "Unbabel/TowerInstruct-7B-v0.2"
checkpoint_name="partial_qe"
cache_dir = "./hf_cache/"

output_dir = "./experiments" + checkpoint_name
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
#print_gpu_memory_every_5secs()

#Data Config
data_files = {}
data_files["train"] = "data/train.ende.json"
data_files["validation"] = "data/valid.ende.json"

dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
src = "src"
tgt = "mt"
label = "label"

prefix = "English:\n"
suffix = "\nGerman:\n "

padding="longest"

print(f"dataset: {dataset['train'][0]}")

# Define filter
def filter_long_text(example):
    example_input = prefix + example['translation'][src] + suffix + example['translation'][tgt]
    encoding = tokenizer(example_input, truncation=False, padding=False)
    return len(encoding['input_ids']) <= 512  # Keep only examples <= 512 tokens

# Apply filter
filtered_dataset = dataset.filter(filter_long_text)
def preprocess_function(examples):
    inputs_full = [prefix + example[src] + suffix + example[tgt] + tokenizer.eos_token for example in examples['translation']]
    encodings_full = tokenizer(inputs_full, return_tensors="pt", padding="longest")

    labels_full = []

    for idx in range(encodings_full.input_ids.shape[0]):
        pad_tokens = torch.sum(encodings_full.input_ids[idx] == tokenizer.pad_token_id)
        examples['translation'][idx]["label"] = ast.literal_eval(examples['translation'][idx]["label"])
        #examples['translation'][idx]["label"] = [x if x!=2 else 1 for x in examples['translation'][idx]["label"]]
        curr_labels = [-100] * len(encodings_full.input_ids[idx])
        curr_labels[-pad_tokens - len(examples['translation'][idx]["label"][1:]): len(curr_labels) - pad_tokens] = examples['translation'][idx]["label"][1:]
        labels_full.append(curr_labels)


    encodings_full['labels'] = torch.tensor(labels_full)

    return encodings_full


processed_datasets = filtered_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
#    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
dev_dataset = processed_datasets["validation"]


peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], inference_mode=False, r=64, lora_alpha=256, lora_dropout=0.1)

# creating model
# Create label mappings
quantization_config=BitsAndBytesConfig(
	load_in_4bit=True,
	llm_int8_threshold=6.0,
	llm_int8_has_fp16_weight=False,
	bnb_4bit_compute_dtype=torch.float16,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4",
)
id2label = {0: "0", 1: "1"}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(model_id, device_map="auto", cache_dir=cache_dir, id2label=id2label,label2id=label2id,num_labels=2) 
model = get_peft_model(model, peft_config)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) 

training_args = TrainingArguments(
        output_dir=output_dir,
        eval_steps=50,
        evaluation_strategy='steps',
        save_steps=50,
        save_strategy='steps',
        logging_steps=2,
        logging_strategy='steps',
        logging_dir=f"{output_dir}/logs",
        learning_rate=1e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=6,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        bf16=True,
        push_to_hub=False,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=checkpoint_name,
        remove_unused_columns=False,
        )

def compute_metrics(pred):
    labels = pred.label_ids

    mask = labels != -100

    filtered_labels = labels[mask]
    filtered_preds = pred.predictions.argmax(axis=-1)[mask]

    accuracy = accuracy_score(filtered_labels, filtered_preds)

   # Calculate precision, recall, and F1-score
    precision = precision_score(filtered_labels, filtered_preds, average='macro')
    recall = recall_score(filtered_labels, filtered_preds, average='macro')
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels").to("cuda")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(model.device))
        #loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95]).to(model.device)) #This worked without distill data
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
        #remove_unused_columns=False,

trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )
trainer.train()

exit()



