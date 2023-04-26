#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import wandb
import re
import pandas as pd
import arrow
import json

import logging
import os
from tqdm.autonotebook import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import random
import sys
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.util import batch_to_device
from ranking_loss import pairwise_hinge_loss, kl_divergence_loss, SmoothAP, SoftBinAP, SupAP, first_egg_loss

from dataclasses import dataclass, field
# from mse_evaluator import MSEEvaluator
# from gpu_debug import ModelLogger
# from reranking_evaluator import CERerankingEvaluator
from torch.utils.data import DataLoader
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import torch
from typing import Dict, Type, Callable, List
import transformers
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from datasets import load_from_disk
from huggingface_hub import PyTorchModelHubMixin


logger = logging.getLogger(__name__)


def split_text_lists(long_texts):
    return [split_texts(long_text) for long_text in long_texts]


def split_texts(long_text):
    """We split long text and record its splitted number
    """
    pieces = re.split(r'\n', long_text)
    pieces = [p.strip() for p in pieces if p]
    return pieces

def split_input_lists(inputs, lengths):
    input_ids, masks = [], []
    start_index = 0
    for length in lengths:
        t_i, t_m = split_inputs(inputs, length, start_index)
        start_index += sum(length)
        input_ids.append(t_i)
        masks.append(t_m)
    return input_ids, masks

def split_inputs(inputs, lengths, start_index=0):
    input_ids, masks = [], []
    for length in lengths:
        input_ids.append(inputs["input_ids"][start_index:start_index + length])
        masks.append(inputs["attention_mask"][start_index:start_index + length])
        start_index += length
    return input_ids, masks

def batch_tokenize(tokenizer, texts, max_length, batch_size=1000):
    all_inputs = {"input_ids": [], "attention_mask": []}
    for start_index in tqdm(range(0, len(texts), batch_size)):
        selected_texts = texts[start_index:start_index+batch_size]
        inputs = tokenizer(selected_texts, padding="max_length", max_length=max_length, truncation=True)
        all_inputs["input_ids"].extend(inputs["input_ids"])
        all_inputs["attention_mask"].extend(inputs["attention_mask"])
    return all_inputs

with open("entity_vocabulary") as f:
    vocabulary = json.loads(f.readlines()[0])

def shrink_entities(hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations):
    hyp_graph_tokens = [vocabulary.index(v.lower()) for v in hyp_graph_tokens]
    annos = list(zip(hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations))
    ori_len = len(annos)
    annos = set(annos)
    if ori_len - len(annos) >= 5:
        print (f"shrinked from {ori_len} to {len(annos)}")
    hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations = [[val[i] for val in annos] for i in range(3)]
    return hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations

    
def preprocess_dataset(tokenizer, example, predict=True):
    PAT = r'\.|\n'
    examples = []
    for ref, ref_graph_tokens, ref_graph_labels, ref_graph_relations, hyps, hyps_graph_tokens, hyps_graph_labels, hyps_graph_relations, trues_graph_tokens, trues_graph_labels, trues_graph_relations, scores, positive in zip(
        example["query"], example["ref_graph_tokens"], example["ref_graph_labels"], example["ref_graph_relations"], example["hyp"], example["hyp_graph_tokens"], example["hyp_graph_labels"], example["hyp_graph_relations"], example["true_graph_tokens"], example["true_graph_labels"], example["true_graph_relations"], example["score"], example["positive"]):
        if len(hyps) != 10 or sum(scores) <= 0.0:
            continue
        new_hyps_graph_tokens, new_hyps_graph_labels, new_hyps_graph_relations = [], [], []
        for hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations in zip(hyps_graph_tokens, hyps_graph_labels, hyps_graph_relations):
            vals = shrink_entities(hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations)
            new_hyps_graph_tokens.append(vals[0])
            new_hyps_graph_labels.append(vals[1])
            new_hyps_graph_relations.append(vals[2])
        hyps_graph_tokens, hyps_graph_labels, hyps_graph_relations = new_hyps_graph_tokens, new_hyps_graph_labels, new_hyps_graph_relations
        indices = range(len(hyps_graph_labels))
        # indices = sorted(range(len(hyps_graph_labels)), key=lambda i: sum(hyps_graph_relations[i]), reverse=True)
        # random.shuffle(indices)
        # print ([set(hyps_graph_tokens[i]) for i in indices if scores[i] == 0.0])
        # indices = indices[:3]
        true_graph_dict = list(zip([vocabulary.index(v.lower()) for v in trues_graph_tokens], trues_graph_labels, trues_graph_relations))
        hyp_graph_dict = list(zip([v for i in indices for v in hyps_graph_tokens[i]], [v for i in indices for v in hyps_graph_labels[i]], [v for i in indices for v in hyps_graph_relations[i]]))
        all_entity_label = [v in true_graph_dict for v in hyp_graph_dict]
        
        examples.append({
            "hyp": [hyps[i] for i in indices], # "label": [1 if not predict and val >= max(scores) - 0.05 else 0 for val in scores][:10],
            "score_label": [scores[i] >= (np.max(scores) - 0.1) for i in indices],
            "label": [scores[i] for i in indices],
            "hyp_entity_labels": [int(v in true_graph_dict) for v in list(hyp_graph_dict)],
            "hyp_graph_tokens": [hyps_graph_tokens[i] for i in indices],
            "hyp_graph_labels": [v for i in indices for v in hyps_graph_labels[i]],
            "ref": ref,
            "hyp_graph_relations": [v for i in indices for v in hyps_graph_relations[i]], 
            "ref_graph_tokens": [vocabulary.index(v.lower()) for v in ref_graph_tokens],
            "ref_graph_labels": ref_graph_labels,
            "ref_graph_relations": ref_graph_relations,
            "true_graph_tokens": [vocabulary.index(v.lower()) for v in trues_graph_tokens],
            "true_graph_labels": trues_graph_labels,
            "true_graph_relations": trues_graph_relations,
        })
        # for hyp, hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations, score in zip(
        #     hyps, hyps_graph_tokens, hyps_graph_labels, hyps_graph_relations, scores):
        #     examples.append({
        #         "hyp": hyp, "label": 1 if not predict and score >= scores[0] else 0,
        #         "hyp_graph_tokens": hyp_graph_tokens, 
        #         "hyp_graph_labels": hyp_graph_labels, 
        #         "ref": ref,
                #"hyp_text": hyp, "ref_text": ref,
        #         "hyp_graph_relations": hyp_graph_relations, 
        #         "ref_graph_tokens": ref_graph_tokens,
        #         "ref_graph_labels": ref_graph_labels,
        #         "ref_graph_relations": ref_graph_relations,
        #     })
    examples = pd.DataFrame(examples)
    examples["hyp"] = examples["hyp"].map(split_text_lists) 
    examples["hyp_lengths"] = examples["hyp"].map(lambda x: [len(v) for v in x]) 
    # examples["hyp_lengths"] = examples["hyp"].map(len)
    # examples["ref_text"] = examples["ref"]
    # examples["ref"] = examples["ref"].map(split_texts) 
    # examples["ref_lengths"] = examples["ref"].map(len) 
    examples["hyp_graph_lengths"] = examples["hyp_graph_tokens"].map(lambda x: [len(v) for v in x]) 
    examples["hyp_graph_sen_lengths"] = examples["hyp_graph_tokens"].map(len)
    examples["ref_graph_lengths"] = examples["ref_graph_tokens"].map(len)

    max_length = 64
    # hyp_input = batch_tokenize(tokenizer, [v for value in examples["hyp"].tolist() for val in value for v in val], max_length=max_length)
    # examples["hyp_input_ids"], examples["hyp_attention_masks"] = split_input_lists(hyp_input, examples["hyp_lengths"])
    # ref_input = batch_tokenize(tokenizer, [v for val in examples["ref"].tolist() for v in val], max_length=max_length)
    # examples["ref_input_ids"], examples["ref_attention_masks"] = split_inputs(ref_input, examples["ref_lengths"])
    # hyp_graph_input = batch_tokenize(tokenizer, [v for value in examples["hyp_graph_tokens"].tolist() for val in value for v in val], max_length=10)
    # examples["hyp_graph_input_ids"], examples["hyp_graph_attention_masks"] = split_input_lists(hyp_graph_input, examples["hyp_graph_lengths"])
    # ref_graph_input = batch_tokenize(tokenizer, [v for val in examples["ref_graph_tokens"].tolist() for v in val], max_length=10)
    # examples["ref_graph_input_ids"], examples["ref_graph_attention_masks"] = split_inputs(ref_graph_input, examples["ref_graph_lengths"])
    if predict:
        examples = examples.to_dict("records")
    else:
        examples = examples.to_dict("lists")
    print (example.shape)
    return examples

def default_data_collator(features):
    start = arrow.now()
    first = features[0]
    batch = {}
    # label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
    dtype = torch.long # if isinstance(label, int) else torch.float
    batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
    batch["true_labels"] = torch.tensor([f["score_label"] for f in features], dtype=torch.float)
    # text_sbank = SentenceBank()
    # graph_sbank = SentenceBank()
    # batch["hyp"] = text_sbank.save_batch([f["hyp"] for f in features])
    # batch["ref"] = text_sbank.save([f["ref"] for f in features])
    # batch["hyp_graph_tokens"] = graph_sbank.save_batch([f["hyp_graph_tokens"] for f in features])
    # batch["ref_graph_tokens"] = graph_sbank.save([f["ref_graph_tokens"] for f in features])
     
    for k, v in first.items():
        if k not in ("label", "label_ids", "score_label") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                if k in ["hyp", "ref", "true_graph_tokens", "true_graph_labels", "true_graph_relations"]: # , "hyp_graph_tokens", "ref_graph_tokens"]:
                    continue
                elif k in ["hyp_input_ids", "hyp_attention_masks", "hyp_graph_input_ids", "hyp_graph_attention_masks", "hyp_graph_tokens"]:
                    batch[k] = torch.tensor(np.concatenate([v for f in features for v in f[k] if v], axis=0))
                elif k in ["ref_graph_input_ids", "ref_graph_attention_masks", "ref_graph_labels", "ref_graph_relations", "hyp_graph_labels", "hyp_graph_relations", "ref_graph_tokens", "hyp_entity_labels"]:
                    try:
                        batch[k] = torch.tensor(np.concatenate([f[k] for f in features if f[k]], axis=0))
                    except Exception as e:
                        print (k, features[0][k])
                        raise e
                else:
                    try:
                        batch[k] = torch.tensor([f[k] for f in features])
                    except Exception as e:
                        print (k, [np.array(f[k]).shape for f in features])
                        raise e

    # for k, v in batch.items():
    #     print (k, v.shape)
    # print ("##############")
    return batch


class SentenceBank:
    """Cache our sentence
    """
    def __init__(self):
        self.s2id = {}
    def flat(self, vals):
        return [v for val in vals for v in val]
    def save_batch(self, texts):
        return torch.cat([self.save(text) for text in texts], dim=0)
    def save(self, texts):
        bank_ids = []
        for text in zip(self.flat(texts)):
            if text not in self.s2id:
                self.s2id[text] = len(self.s2id)
            bank_ids.append(self.s2id[text])
        return torch.tensor(bank_ids)

class EntityCrossEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name_or_path, device=None, max_seq_length=None, dropout=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        self.text_model = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            )
        embeddings_dimension = self.text_model.config.hidden_size
        
        self._target_device = device
        self.entity_label_layer = nn.Embedding(4, embeddings_dimension)
        self.entity_token_layer = nn.Embedding(10000, embeddings_dimension)
        self.entity_label_layer.require_grad = True
        self.relation_layer = nn.Embedding(2, embeddings_dimension)
        self.relation_layer.require_grad = True
        self.cross_attn = nn.MultiheadAttention(embeddings_dimension, 4, batch_first=True)
        self.fuse_attn = nn.MultiheadAttention(embeddings_dimension, 4, batch_first=True)
        self.cross_dropout = nn.Dropout(p=dropout)
        self.fuse_dropout = nn.Dropout(p=dropout)
        self.embeddings_dimension = embeddings_dimension
        self.output_layer = nn.Linear(embeddings_dimension, 1)
        self.number_layer = nn.Linear(embeddings_dimension, 1)
        self.global_embeddings = nn.Parameter(torch.tensor(torch.ones(embeddings_dimension), requires_grad=True))
        self.activation = nn.Sigmoid()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, text_ids, input_ids, attention_masks, batch_size=64, if_sum=True):
        start = arrow.now()
        flat_text_ids = torch.cat(text_ids)
        text_lengths = torch.tensor([len(val) for val in text_ids], dtype=torch.long)
        # flat_text_ids = [v.item() for texts in text_ids for v in texts]
        # tids = list(set(flat_text_ids))
        uniques, text_ids, counts = torch.unique(flat_text_ids, sorted=True, return_inverse=True, return_counts=True)
        _, text_id_sorted = torch.sort(text_ids, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(cum_sum.device), cum_sum[:-1]))
        first_indexes = text_id_sorted[cum_sum]
        text_ids = torch.tensor_split(text_ids, torch.cumsum(text_lengths, dim=0))[:-1]
        # print (f"    Distinct cost {(arrow.now() - start).total_seconds()}s")
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        select_input_ids = input_ids[first_indexes]
        select_attention_masks = attention_masks[first_indexes]
        start = arrow.now()
        all_embeddings = []
        for start_index in trange(0, len(select_input_ids), batch_size, desc="Batches", disable=True):
            batch_inputs = select_input_ids[start_index:start_index+batch_size]
            batch_masks = select_attention_masks[start_index:start_index+batch_size]
            out_features = self.text_model(input_ids=batch_inputs, attention_mask=batch_masks, return_dict=False)[0]
            if if_sum:
                embeddings = self.mean_pooling(out_features, batch_masks)
            else:
                embeddings = out_features
            all_embeddings.extend(embeddings)
        # print (f"    Batch encode cost {(arrow.now() - start).total_seconds()}s")
        start = arrow.now()
        # all_embeddings = dict(zip(all_inputs["texts"], all_embeddings))
        # all_embeddings = # [all_embeddings[:len(text_ids[0])], all_embeddings[len(text_ids[0]):]]
        all_embeddings = torch.stack(all_embeddings)
        all_embeddings = [
            all_embeddings[texts] for texts in text_ids]
        # print (f"    Encoding stack cost {(arrow.now() - start).total_seconds()}s")
        return all_embeddings

    def generate_token_embeddings(self, all_embeddings, texts):
        embeddings = [all_embeddings[sen] for sen in texts]
        embeddings = torch.stack(embeddings)
        return embeddings

    def forward(self,
        hyp=None,
        ref=None,
        hyp_input_ids=None,
        ref_input_ids=None,
        hyp_attention_masks=None,
        ref_attention_masks=None,
        hyp_lengths=None,
        hyp_sen_lengths=None,
        ref_lengths=None,
        hyp_graph_tokens=None,
        ref_graph_tokens=None,
        hyp_graph_input_ids=None,
        ref_graph_input_ids=None,
        hyp_graph_attention_masks=None,
        ref_graph_attention_masks=None,
        hyp_graph_lengths=None,
        ref_graph_lengths=None,
        hyp_graph_labels=None,
        hyp_graph_relations=None,
        ref_graph_labels=None,
        ref_graph_relations=None,
        prediction=False,
        labels=None,
        output_attentions=True,
    ):
        start = arrow.now()
        # hyps, refs = self.encode_text([hyp, ref], [hyp_input_ids, ref_input_ids], [hyp_attention_masks, ref_attention_masks])
        # print (hyp.shape, hyp_input_ids.shape, hyp_attention_masks.shape)
        # hyps = self.encode_text([hyp], [hyp_input_ids], [hyp_attention_masks])[0]
        # print (hyps.shape, hyp_lengths, hyp_sen_lengths)
        # text_embeddings = self.encode_text(text_input_ids, text_attention_masks)
        batch_size, hyp_num = hyp_lengths.shape
        # hyps = gather_embeddings(hyps, hyp_lengths.reshape(-1))
        # hyps = torch.ones(batch_size * hyp_num, 768).to(hyp.device)
        # print (f"Hyp Encode cost {(arrow.now() - start).total_seconds()}s")
        start = arrow.now()
        # refs = gather_embeddings(refs, ref_lengths)
        # refs = torch.zeros_like(hyps).to(hyps.device)
        # # # print (f"Ref Encode cost {(arrow.now() - start).total_seconds()}s")
        # hyp_graph_tokens, ref_graph_tokens = self.encode_text([hyp_graph_tokens, ref_graph_tokens], [hyp_graph_input_ids, ref_graph_input_ids], [hyp_graph_attention_masks, ref_graph_attention_masks], if_sum=True)
        hyp_graph_tokens = self.entity_token_layer(hyp_graph_tokens)
        ref_graph_tokens = self.entity_token_layer(ref_graph_tokens)
        # hyp_graph_tokens = gather_embeddings(hyp_graph_tokens, hyp_graph_lengths)
        # ref_graph_tokens = gather_embeddings(ref_graph_tokens, ref_graph_lengths)
        # print (f"Graph Encode cost {(arrow.now() - start).total_seconds()}s")
        start = arrow.now()
        hyp_label_embeddings = self.entity_label_layer(hyp_graph_labels)
        hyp_relation_embeddings = self.relation_layer(hyp_graph_relations)
        ref_label_embeddings = self.entity_label_layer(ref_graph_labels)
        ref_relation_embeddings = self.relation_layer(ref_graph_relations)
        hyp_entity_embeddings = hyp_graph_tokens + hyp_label_embeddings + hyp_relation_embeddings
        ref_entity_embeddings = ref_graph_tokens + ref_label_embeddings + ref_relation_embeddings
        hyp_entity_embeddings = split_embeddings(hyp_entity_embeddings, hyp_graph_lengths.reshape(-1))
        ref_entity_embeddings = split_embeddings(ref_entity_embeddings, ref_graph_lengths.reshape(-1))
        hyp_entity_embeddings, hyp_mask = pad_tensor_list(hyp_entity_embeddings)
        ref_entity_embeddings, ref_mask = pad_tensor_list(ref_entity_embeddings)
        global_embeddings = self.global_embeddings.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        ref_mask = ref_mask.repeat_interleave(4, dim=0)
        global_embeddings, cross_attn = self.cross_attn(global_embeddings, ref_entity_embeddings, ref_entity_embeddings, attn_mask=(ref_mask-1e-9))
        global_embeddings = global_embeddings.repeat_interleave(hyp_num, dim=0)
        global_number = self.number_layer(global_embeddings).squeeze(-1).squeeze(-1)
        # ref_entity_embeddings = ref_entity_embeddings.repeat_interleave(hyp_num, dim=0)
        # ref_mask = ref_mask.repeat_interleave(hyp_num, dim=0)
        # hyp_cross_entity_embeddings, cross_attn = self.cross_attn(hyp_entity_embeddings, ref_entity_embeddings, ref_entity_embeddings)
        # hyp_entity_embeddings = hyp_entity_embeddings + self.cross_dropout(hyp_cross_entity_embeddings)
        hyp_entity_embeddings = hyp_entity_embeddings + global_embeddings
        hyp_mask = hyp_mask.squeeze(1) + 1e-30
        hyp_entity_scores = self.activation(9 * self.output_layer(hyp_entity_embeddings)).squeeze(-1)
        # hyp_entity_scores = self.output_layer(hyp_entity_embeddings).squeeze(-1)
        # hyp_entity_scores = (torch.sign(hyp_entity_scores) + 1) / 2.0
        # score = 2 * (hyp_entity_scores * hyp_mask).sum(-1)/ (hyp_mask.sum(-1) + global_number)
        # embeddings = hyp_entity_embeddings
        # embeddings, fuse_attn = self.fuse_attn(hyps.unsqueeze(1), hyp_entity_embeddings, hyp_entity_embeddings, attn_mask=hyp_mask)
        # ref_embeddings, cross_attn = self.cross_attn(hyps.unsqueeze(1), ref_entity_embeddings, ref_entity_embeddings, attn_mask=ref_mask)
        # embeddings = hyps.unsqueeze(1) + self.fuse_dropout(embeddings)
        # embeddings = self.fuse_dropout(embeddings)
        # ref_embeddings = self.cross_dropout(ref_embeddings)
        # embeddings = embeddings.reshape(batch_size, hyp_num, -1)
        # ref_embeddings = ref_embeddings.reshape(batch_size, hyp_num, -1)
        # score = torch.bmm(embeddings, global_embeddings.permute(0, 2, 1))
        # embeddings = torch.stack(fused_hyp_entity_embeddings)
        # hyp_entity_embeddings = gather_embeddings(hyp_entity_embeddings, hyp_graph_lengths)
        # ref_entity_embeddings = gather_embeddings(ref_entity_embeddings, ref_graph_lengths)

        # hyps = hyps + hyp_entity_embeddings
        # refs = refs + ref_entity_embeddings
        # embeddings = torch.cat([embeddings, ref_embeddings], dim=2)
        # score = self.output_layer(embeddings)

        # score = self.activation(score).squeeze(-1)
        # score = score.squeeze(-1)
        # score = score / 1.0
        # score = nn.functional.gumbel_softmax(score, tau=0.5, hard=False)
        # if prediction:
        #     score = self.activation(score)
        # print (f"Model cost {(arrow.now() - start).total_seconds()}s")
        # score = score.reshape(batch_size, hyp_num)
        score = hyp_entity_scores
        if output_attentions:
            print (score.shape, cross_attn.shape, fuse_attn.shape)
            return score, cross_attn, fuse_attn
        else:
            return score


def pad_tensor_list(tensor_list, pad_value=0):
    max_length = max(tensor.shape[0] for tensor in tensor_list)
    padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[0]), value=pad_value) for tensor in tensor_list]
    mask = torch.zeros((len(tensor_list), max_length), dtype=torch.float)
    for i, tensor in enumerate(tensor_list):
        mask[i, :tensor.shape[0]] = 1
    # mask = (1 - mask) * -1e9
    # mask = torch.repeat_interleave(mask, 4, dim=0)
    mask = mask.unsqueeze(1).to(tensor_list[0].device)
    return torch.stack(padded_tensors), mask


def gather_selected_embeddings(embeddings, text_ids, lengths):
    select_embeddings = torch.index_select(embeddings, 0, text_ids)
    return gather_embeddings(select_embeddings, lengths) 

def split_embeddings(embeddings, lengths):
    embeddings = torch.tensor_split(embeddings, torch.cumsum(lengths, dim=0).cpu())[:-1]
    return embeddings

def gather_embeddings(embeddings, lengths):
    embeddings = torch.tensor_split(embeddings, torch.cumsum(lengths, dim=0).cpu())[:-1]
    embeddings = torch.stack([val.sum(dim=0) for val in embeddings], dim=0)
    return embeddings


class ScorerTrainer(Trainer):
    loss_act = SupAP()
        # self.model = EntityEnrichEmbedding(768, text_model_name, self._target_device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        entity_labels = inputs.get("hyp_entity_labels")
        outputs = model(
            hyp=inputs.get("hyp"),
            ref=inputs.get("ref"),
            hyp_input_ids=inputs.get("hyp_input_ids"),
            ref_input_ids=inputs.get("ref_input_ids"),
            hyp_attention_masks=inputs.get("hyp_attention_masks"),
            ref_attention_masks=inputs.get("ref_attention_masks"),
            hyp_lengths=inputs.get("hyp_lengths"),
            hyp_sen_lengths=inputs.get("hyp_sen_lengths"),
            ref_lengths=inputs.get("ref_lengths"),
            hyp_graph_tokens=inputs.get("hyp_graph_tokens"),
            ref_graph_tokens=inputs.get("ref_graph_tokens"),
            hyp_graph_input_ids=inputs.get("hyp_graph_input_ids"),
            ref_graph_input_ids=inputs.get("ref_graph_input_ids"),
            hyp_graph_attention_masks=inputs.get("hyp_graph_attention_masks"),
            ref_graph_attention_masks=inputs.get("ref_graph_attention_masks"),
            hyp_graph_lengths=inputs.get("hyp_graph_lengths"),
            ref_graph_lengths=inputs.get("ref_graph_lengths"),
            hyp_graph_labels=inputs.get("hyp_graph_labels"),
            hyp_graph_relations=inputs.get("hyp_graph_relations"),
            ref_graph_labels=inputs.get("ref_graph_labels"),
            ref_graph_relations=inputs.get("ref_graph_relations"),
            output_attentions=False,
        )
        # print (outputs.shape, entity_labels.shape)
        # self.loss_act = self.loss_act.to(outputs.device)
        # loss_act = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]).to(outputs.device))
        loss_act = nn.BCEWithLogitsLoss()
        loss = loss_act(outputs.squeeze(-1), entity_labels)
        # loss = kl_divergence_loss(outputs.squeeze(-1), labels)
        # loss = self.loss_act(outputs.squeeze(-1), labels)
        # loss = first_egg_loss(outputs.squeeze(-1), labels)
        # loss = nn.MSELoss()(outputs.squeeze(-1), labels)
        # mape = torch.abs(outputs.squeeze(-1) - labels)
        # loss = torch.mean(mape)
        if return_outputs:
            return loss, {"logits": outputs}
        return loss


    def predict(self, dataloader,
               batch_size: int = 256,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = dataloader
        dataloader.collate_fn = self.smart_batching_collate
        if show_progress_bar:
            iterator = tqdm(dataloader, desc="Batches")

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for hyps, refs, hyps_graph_tokens, hyps_graph_labels, hyps_graph_lengths, hyps_graph_relations, refs_graph_tokens, refs_graph_labels, refs_graph_lengths, refs_graph_relations, labels in iterator:
                logits = self.model(hyps, refs, hyps_graph_tokens, hyps_graph_labels, hyps_graph_lengths, hyps_graph_relations, refs_graph_tokens, refs_graph_labels, refs_graph_lengths, refs_graph_relations)
                pred_scores.extend(logits)
        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        return pred_scores


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_from_disk(data_args.dataset_name)
    test_datasets = load_from_disk(data_args.test_dataset_name)

    # Labels
    num_labels = 1

    
    model = EntityCrossEncoder(model_args.model_name_or_path, model_args.cache_dir)
    
    # model = CrossEncoder(model_args.model_name_or_path)

    # Preprocessing the raw_datasets

    if training_args.do_train:
        train_raw_dataset = raw_datasets
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_raw_dataset.map(
                lambda x: preprocess_dataset(model.tokenizer, x, False),
                batched=True,
                num_proc=8,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_raw_dataset.column_names,
                desc="Running preprocessing on train dataset",
            ).shuffle()
        # train_dataset = preprocess_dataset(train_raw_dataset)

    if training_args.do_eval:
        # eval_raw_dataset = raw_datasets["validation"]
        eval_raw_dataset = test_datasets
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_raw_dataset.map(
                lambda x: preprocess_dataset(model.tokenizer, x, False),
                batched=True,
                num_proc=8,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_raw_dataset.column_names,
                desc="Running preprocessing on evaluation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_raw_dataset = test_datasets
        predict_dataset = preprocess_dataset(predict_raw_dataset)

    def mrr_at_k(true_scores, pred_scores, k):
        # Get the indices of the top-k predictions
        top_k_indices = np.argsort(-pred_scores, axis=1)[:, :k]
    
        # Select only the first k columns of true_scores
        true_scores = true_scores[np.arange(true_scores.shape[0])[:, np.newaxis], top_k_indices]
    
        # Compute the cumulative sum of true_scores being 0 or less
        cumsum = np.cumsum(true_scores <= 0, axis=1)
    
        # Compute the reciprocal rank for each row
        reciprocal_ranks = 1. / (np.arange(1, k + 1) + cumsum)
    
        # Return the mean reciprocal rank over all rows
        return np.mean(reciprocal_ranks)

    def get_mean_true_score(true_scores, pred_scores):
        pred_index = np.argmax(pred_scores, axis=1)
        true_index = np.argmax(true_scores, axis=1)
        corresponding_true_scores = true_scores[np.arange(pred_scores.shape[0]), pred_index]
        wrong_index = np.argmin(corresponding_true_scores - np.max(true_scores, axis=1))
        # print (true_scores[wrong_index], pred_scores[wrong_index], wrong_index, corresponding_true_scores[wrong_index], np.max(true_scores, axis=1)[wrong_index], eval_dataset[int(wrong_index)])
        return np.mean(corresponding_true_scores), wandb.Histogram(corresponding_true_scores), wandb.Histogram(pred_index), wandb.Histogram(corresponding_true_scores - np.max(true_scores, axis=1)), wandb.Histogram(true_index), np.mean(corresponding_true_scores - np.max(true_scores, axis=1))


    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        # f1_metric = evaluate.load("bstrai/classification_report")
        # results = f1_metric.compute(predictions=1 * (preds >= 0.5), references=p.label_ids[2])
        # return results
        mrrs = {f"mrr@{k}": mrr_at_k(p.label_ids[2], preds, k) for k in [1]}
        mrrs["rad_graph_partial"], mrrs["select_scores"], mrrs["pred_index"], mrrs["best_gap"], mrrs["true_index"], mrrs["mean_gap"] = get_mean_true_score(p.label_ids[2], preds) 
        mrrs["lowest_candiate"] = np.mean(np.min(p.label_ids[2], axis=1))
        mrrs["highest_candiate"] = np.mean(np.max(p.label_ids[2], axis=1))
        return mrrs
        return {"accuracy": ((1.0 * (preds >= 0.5)) == p.label_ids[2]).astype(np.float32).mean().item()}

    trainer = ScorerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,# .select(list(range(10000))) if training_args.do_train else None,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    print (eval_dataset)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    if training_args.push_to_hub:
        model.save_to_hub("mimic-scorer")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    wandb.init()
    main()
