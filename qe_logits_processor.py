import transformers
import copy
from transformers import LogitsProcessor
import argparse
import os
import sys
import torch
import numpy as np
from torch import nn
import logging
import string

class QELogitsProcessor(LogitsProcessor):
    def __init__(self, qe_model, tokenizer, prompt, topk, num_beams, alpha):
        self.qe_model = qe_model # Adapted Token-level QE model
        self.tokenizer = tokenizer # Tokenizer that is same for NMT and QE

        self.prompt = prompt # Prompt for Token-level QE

        self.topk = topk # How many top tokens in each beam
        self.num_beams = num_beams # How many beams is the decoding with
        self.alpha = alpha # re-ranking Weight

        self.step = 0
        self.debug = True
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        topk_values, topk_indices = torch.topk(scores, self.topk, dim=1)
        next_tokens = topk_indices.flatten().reshape(self.topk*self.num_beams,1).to(self.qe_model.device)  #Topk items for each beam joine
        attention_mask = torch.ones(self.topk*self.num_beams,1).to(self.qe_model.device)

        new_inputs = {"input_ids": next_tokens, "attention_mask": attention_mask}


        if self.step == 0:
            self.prefix_len = input_ids.shape[1]
            inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).to(self.qe_model.device)
            self.past_key_values = self.qe_model(**inputs, use_cache=True).past_key_values
            self.prompt_tokens = inputs['input_ids'].to(self.qe_model.device)
            self.prev_tokens = torch.cat((self.prompt_tokens, next_tokens), dim=1)

        else:
            curr_hyps = input_ids[:,self.prefix_len:].repeat_interleave(self.topk, dim=0).to(self.qe_model.device)
            curr_hyps = torch.cat((curr_hyps, next_tokens), dim=1)
            self.curr_tokens = torch.cat((self.prompt_tokens, curr_hyps), dim=1)
            #self.prompt_tokens_full = torch.cat((self.prompt_tokens, input_ids[:,s]))

            beam_align_idx = self.beam_align(self.prev_tokens, self.curr_tokens)
            #beam_align_idx = [x for x in beam_align_idx for _ in range(self.topk)]
            if len(beam_align_idx) == self.num_beams * self.topk:
                self.reorder_cache(beam_align_idx)
            else:
                self.past_key_values = None# This is a scenario where we saw eos token and caching needs to be updated as the prev tokens and curr tokens dont match. Current fix to reupdate cache is hacky
            #self.prev_tokens = torch.cat((new_inputs_prev, next_tokens), dim=1)
            self.prev_tokens = self.curr_tokens


        qe_outputs = self.qe_model(**new_inputs, use_cache=True, past_key_values=self.past_key_values)


        qe_scores = torch.nn.functional.log_softmax(qe_outputs.logits, dim=-1)[:,0,0] # Class 0 is the likelihood that each token is good
        qe_labels = torch.argmax(qe_outputs.logits, dim=-1).squeeze()

        self.past_key_values = qe_outputs.past_key_values

        del qe_outputs



        qe_scores = torch.tensor(qe_scores).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)

        scores= torch.nn.functional.log_softmax(scores, dim=-1)
        reranked_scores = torch.full_like(scores, -float("inf")).to(input_ids.device)

        alphas = torch.ones(self.topk*self.num_beams,1) * self.alpha
        alphas = alphas.to(input_ids.device).view(input_ids.shape[0],-1)

        for i in range(reranked_scores.shape[0]):
            reranked_scores[i,topk_indices[i,:]] = alphas[i]*scores[i,topk_indices[i,:]] + (1 - alphas[i])*qe_scores[i]

        if self.debug:
            curr_hyp = self.tokenizer.batch_decode(input_ids[:,self.prefix_len:], skip_special_tokens=False)
            for hyp in curr_hyp:
                print(hyp)
            print("*"*50)
            print("*"*50)


        self.step+=1

        if self.step == 1: # This is the first step where we only have 1 beam. We start merging after the num_beams are initialized with tokens at the first step
            return scores


        return reranked_scores

    def reorder_cache(self, beam_align_idx):

        self.past_key_values = list(self.past_key_values)

        for layer_id in range(len(self.past_key_values)):
            self.past_key_values[layer_id] = list(self.past_key_values[layer_id])

            self.past_key_values[layer_id][0] = self.past_key_values[layer_id][0][beam_align_idx]
            self.past_key_values[layer_id][1] = self.past_key_values[layer_id][1][beam_align_idx]
            self.past_key_values[layer_id] = tuple(self.past_key_values[layer_id])

        self.past_key_values = tuple(self.past_key_values)
        return


    def beam_align(self, tensor_old, tensor_new):

        row_mapping = [] # First elem indicates which row tensor_new corresponds with old

        for i in range(tensor_new.size(0)):
            for j in range(tensor_old.size(0)):
                if torch.all(tensor_old[j,:] == tensor_new[i,:-1]):
                    row_mapping.append(j)
                    break

        return row_mapping
