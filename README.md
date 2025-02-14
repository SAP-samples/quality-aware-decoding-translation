# Quality-Aware Decoding: Unifying Quality Estimation and Decoding
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2502.08561-b31b1b.svg)](https://arxiv.org/abs/2502.08561)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/quality-aware-decoding-translation)](https://api.reuse.software/info/github.com/SAP-samples/quality-aware-decoding-translation)

## Description
This repository contains Quality-aware decoding code from this [paper](https://arxiv.org/abs/2502.08561). You can use the code here to re-train our partial Quality Estimation models and integrate them to perfrom Quality-Aware Decoding.

## Abstract

Quality Estimation (QE) models for Neural Machine Translation (NMT) predict the quality of the hypothesis without having access to the reference.
An emerging research direction in NMT involves the use of QE models, which have demonstrated high correlations with human judgment and can enhance translations through Quality-Aware Decoding. Although several approaches have been proposed based on sampling multiple candidate translations and picking the best candidate, none have integrated these models directly into the decoding process. In this paper, we address this by proposing a novel token-level QE model capable of reliably scoring partial translations. We build a uni-directional QE model for this, as decoder models are inherently trained and efficient on partial sequences. We then present a decoding strategy that integrates the QE model for Quality-Aware decoding and demonstrate that the translation quality improves when compared to the N-best list re-ranking with state-of-the-art QE models (up to $1.39$ XCOMET-XXL $\uparrow$). Finally, we show that our approach provides significant benefits in document translation tasks, where the quality of N-best lists is typically suboptimal.

## Requirements
We only rely on the transformers library for fusing models. However, to make the inference possible on 1 GPU and load fine-tuned models, peft and bitsandbytes are also required.

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Peft](https://github.com/huggingface/peft)

## Download and Installation

Only installing the above libraries is necessary. No other installation is required except cloning the git repository.

Clone this repository
```
git clone https://github.com/SAP-samples/quality-aware-decoding-translation
cd quality-aware-decoding-translation
```

## Example usage of Quality-Aware Decoding

```
from transformers import  AutoModelForCausalLM, AutoModelForTokenClassification, LogitsProcessorList, AutoTokenizer
from qe_logits_processor import QELogitsProcessor

model_id = "Unbabel/TowerInstruct-7B-v0.2"
cache_dir = "./hf_cache/"

tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
tokenizer.padding_side = "left" # Inference mode
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to("cuda:0")

id2label = {0: "0", 1: "1"}
label2id = {v: k for k, v in id2label.items()}
qe_model = AutoModelForTokenClassification.from_pretrained(model_id,cache_dir=cache_dir, id2label=id2label,label2id=label2id, device_map="auto",  max_memory={0: "0GB", 1: "48GB"})

qe_model.load_adapter("skoneru/tower_qe_ende_mqm") 

prefix = "<|im_start|>user\nTranslate the sentence from English into German.\nEnglish: "
suffix = "\nGerman:"
pe_suffix = "<|im_end|>\n<|im_start|>assistant\n"


prefix_classifier = "English:\n"
suffix_classifier = "\nGerman:\n"

padding="longest"

src_batch = ["Department of Homeland Security."]
num_return_sequences=5
num_beams=5
topk=5
alpha=0.5


text = [prefix + src_entry + suffix + pe_suffix for src_entry in src_batch]
inputs = tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

text_classifier = [prefix_classifier + src_entry + suffix_classifier for src_entry in src_batch]
text_classifier = [x for x in text_classifier for _ in range(num_beams*topk)]

logits_processor = LogitsProcessorList(
    [
    QELogitsProcessor(qe_model=qe_model, tokenizer=tokenizer, prompt=text_classifier, topk=topk, num_beams=num_beams, alpha=alpha),
    ]
)


### Without Quality-Aware Decoding
output = model.generate(**inputs, num_beams=num_beams, max_new_tokens=1024, eos_token_id =32005, num_return_sequences=num_return_sequences)
print("Quality-Unaware Decoding Top Sequences:")
for idx in range(len(src_batch)):
    for entry in range(num_return_sequences):
        curr_hyp = tokenizer.decode(output[idx*num_return_sequences + entry][len(inputs.input_ids[idx]):], skip_special_tokens=True)
        print(curr_hyp)



### With Quality-Aware Decoding
output = model.generate(**inputs, num_beams=num_beams, max_new_tokens=1024, eos_token_id =32005, num_return_sequences=num_return_sequences, logits_processor=logits_processor)
print("Quality-Aware Decoding Top Sequences:")
for idx in range(len(src_batch)):
    for entry in range(num_return_sequences):
        curr_hyp = tokenizer.decode(output[idx*num_return_sequences + entry][len(inputs.input_ids[idx]):], skip_special_tokens=True)
        print(curr_hyp)
```

We showed in the paper that the Tower and Tower QE can be fused at the decoding time. To reproduce this on WMT 23, you can use the following script with default parameters:

```
python quality_aware_tower.py
```

Note that the amount of beams encoded at the QE model is num_beams*topk. Hence, it would need a lot more memory as we also do not perform any quantization. In our setup, we use 4 NVIDIA A6000 GPU's where we use the latter 3 only for the QE model. This was necessary as the WMT'23 English to German was paragraph-level and hence needing more memory.

You can change the following variables to control the decoding

```
topk=5 # How many top tokens for each beam
alpha=0.5 # The re-ranking weight
num_return_sequences=5 # How many top sequences you want from the joint decoding
num_beams=5 # How many beams during decoding. Effective beams used for QE is num_beams*topk
```

## How to use the Partial Quality Estimation Models

We provide the two Quality Estimation models for English to German and Chinese to English. The models are fine-tuned on Tower with additional LoRA parameters and classification head. Hence, we only provide the weights for them. The script will automatically download the base model [Tower-Instruct V2](https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2) to the cache directory from huggingface.

You can find the models here:


English to German: [skoneru/tower_qe_ende_mqm](https://huggingface.co/skoneru/tower_qe_ende_mqm)

Chinese to English: [skoneru/tower_qe_zhen_mqm](https://huggingface.co/skoneru/tower_qe_zhen_mqm)

If you would like to use the QE models for evaluating quality, you can refer to the following script:

```
eval_qe.py
```

The script needs to be modified for pointing to the right filepaths. Please change the following variables

    1. src: File path to the source file
    2. mt: File path to the hypothesis file
    3. num_return_sequences: We assume that the source and hypothesis are aligned. 
        However, each source can be translated with multiple systems or the N-best list. 
        Therefore set this variable to other than 1 if the same source is translated multiple times.
        Order the file such that the first num_return_sequences are the hypothesis of the first source
    4. output_file: File path to where the scores should be saved

## How to reproduce the Partial Quality Estimation Models

We provide the fine-tuning script and the processed data to reproduce building the QE models. In the `data` folder, we provide the data in the json format. It includes the source, hypothesis and the labels of the hypothesis sentence according to Tower Tokenizer. If you would like to use another model, you would need to re-map these labels.

Next, you can use this data to fine-tune Tower by adding LoRA and the classification head. You can use the following script for this:

```
finetune_towerqe.py
```

## Known Issues

1. Out-Of-Memory: If you set big topk or num_beams, then the QE model may go OOM. You atleast need 2 GPU's and may need more if you are translating long sequences. If you have multiple GPU's, then you can mention them when loading the model. Otherwise, you can try enhancements such as loading in 8-bit or 4-bit. However, we did not try them and do not know how much the performance gap is.

## Citations
If you use this code in your research or want to refer to our work, please cite:

```
@@misc{koneru2025qualityawaredecodingunifyingquality,
      title={Quality-Aware Decoding: Unifying Quality Estimation and Decoding}, 
      author={Sai Koneru and Matthias Huck and Miriam Exel and Jan Niehues},
      year={2025},
      eprint={2502.08561},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08561}, 
}
```
#### Authors:
 - Sai Koneru
 - Matthias Huck
 - Miriam Exel
 - Jan Niehues


## How to obtain support
[Create an issue](https://github.com/SAP-samples/quality-aware-decoding-translation/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
