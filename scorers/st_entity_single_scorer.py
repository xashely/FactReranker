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
import nltk  # Here to have a nice missing dependency error message early on
import re
import pandas as pd
import arrow
import json

import logging
import os
from tqdm.autonotebook import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
import random
import sys
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.util import batch_to_device
# from ranking_loss import pairwise_hinge_loss, kl_divergence_loss, SmoothAP, SoftBinAP, SupAP, first_egg_loss

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

# with open("entity_vocabulary") as f:
#     vocabulary = json.loads(f.readlines()[0])
entity_labels = ["[ANAT-DP]", "[OBS-DP]", "[OBS-U]", "[OBS-DA]"]
relation_labels = ["[NA]", "[REL]"]


def build_hyp_dicts(examples):
    hyp_dicts = []
    for example in examples:
        hyp_dicts.append([])
        tokens, labels, relations = example.get("hyp_graph_tokens"), example.get("hyp_graph_labels"), example.get("hyp_graph_relations")
        for ht, hl, hr in zip(tokens, labels, relations):
             hyp_dicts[-1].append(list(zip(ht, hl, hr)))
    return hyp_dicts

def preprocess_dataset(example, predict=True, tokenizer=None, data_args=None):
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    inputs, targets = [], []
    entity_prefix = "[ENT]"

    for ref, ref_input_ids, ref_graph_tokens, ref_graph_labels, ref_graph_relations, hyps, hyps_graph_tokens, hyps_graph_labels, hyps_graph_relations, hyps_indexes, trues_graph_tokens, trues_graph_labels, trues_graph_relations, scores, positive in zip(
        example["query"], example["ref_index"], example["ref_graph_tokens"], example["ref_graph_labels"], example["ref_graph_relations"], example["hyp"], example["hyp_graph_tokens"], example["hyp_graph_labels"], example["hyp_graph_relations"], example["hyp_index"], example["true_graph_tokens"], example["true_graph_labels"], example["true_graph_relations"], example["score"], example["positive"]):
    
        ref_graph_dict = list(zip(ref_graph_tokens, ref_graph_labels, ref_graph_relations))
        true_graph_dict = list(zip(trues_graph_tokens, trues_graph_labels, trues_graph_relations))
        source = f" {entity_prefix} ".join([" ".join([v[0], entity_labels[v[1]], relation_labels[v[2]]]) for v in ref_graph_dict])
        target = f" {entity_prefix} ".join([" ".join([v[0], entity_labels[v[1]], relation_labels[v[2]]]) for v in true_graph_dict])
        inputs.append(source)
        targets.append(target)
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding="max_length", truncation=True)

    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

       
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
    max_source_length: Optional[int] = field(
        default=360,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    max_target_length: Optional[int] = field(
        default=360,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    special_tokens_dict = {'additional_special_tokens': ["[ANAT-DP]", "[OBS-DP]", "[OBS-U]", "[OBS-DA]", "ENT]", "[NA]", "[REL]"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

        
    # Preprocessing the raw_datasets

    if training_args.do_train:
        train_raw_dataset = raw_datasets
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_raw_dataset.map(
                lambda x: preprocess_dataset(x, tokenizer=tokenizer, data_args=data_args),
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
                lambda x: preprocess_dataset(x, tokenizer=tokenizer, data_args=data_args),
                batched=True,
                num_proc=8,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_raw_dataset.column_names,
                desc="Running preprocessing on evaluation dataset",
            )

    if training_args.do_predict:
        predict_raw_dataset = test_datasets
        predict_dataset = predict_raw_dataset.map(
            lambda x: preprocess_dataset(x, tokenizer=tokenizer, data_args=data_args),
            batched=True,
            num_proc=8,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=train_raw_dataset.column_names,
            desc="Running preprocessing on test dataset",
        )
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    metric = evaluate.load("rouge")
    # metric_bert = evaluate.load("bertscore")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        ml = max(preds.shape[1], labels.shape[1])
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
        # print (preds[0], decoded_preds[0], decoded_labels[0])

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print (decoded_preds[0], decoded_labels[0])

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # result_bert = metric_bert.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        # result_bert = {"bertscore_f1":round(v, 4) for v in result_bert["f1"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        # result.update(result_bert)
        return result
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,# .select(list(range(10000))) if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
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

    def get_mean_true_score(true_scores, pred_scores):
        pred_index = np.argmax(pred_scores, axis=1)
        true_index = np.argmax(true_scores, axis=1)
        corresponding_true_scores = true_scores[np.arange(pred_scores.shape[0]), pred_index]
        wrong_index = np.argmin(corresponding_true_scores - np.max(true_scores, axis=1))
        # print (true_scores[wrong_index], pred_scores[wrong_index], wrong_index, corresponding_true_scores[wrong_index], np.max(true_scores, axis=1)[wrong_index], eval_dataset[int(wrong_index)])
        return np.mean(corresponding_true_scores), wandb.Histogram(corresponding_true_scores), wandb.Histogram(pred_index), wandb.Histogram(corresponding_true_scores - np.max(true_scores, axis=1)), wandb.Histogram(true_index), np.mean(corresponding_true_scores - np.max(true_scores, axis=1))

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        predict_datasets = predict_dataset
        entity_label_dict = dict(zip(entity_labels, range(len(entity_labels))))
        relation_label_dict = dict(zip(relation_labels, range(len(relation_labels))))
        
        hyp_dicts = build_hyp_dicts(test_datasets)

        # predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_datasets, metric_key_prefix="predict").predictions
        def clean_prediction(prediction):
            prediction = prediction.split("</s>")[0].split("<s>")[1]
            prediction = prediction.split(" [ENT] ")
            prediction = [val.split() for val in prediction]
            new_prediction = []
            for val in prediction:
                if len(val) != 3:
                    continue
                new_prediction.append((val[0], entity_label_dict.get(val[1], 4), relation_label_dict.get(val[2], 2)))
            return new_prediction
        predictions = tokenizer.batch_decode(predictions[0])
        predictions = [clean_prediction(val) for val in predictions]
        scores = []
        true_scores = [val["score"] for val in test_datasets]
        for hyp_dict, prediction in zip(hyp_dicts, predictions):
            scores.append([])
            for hyp in hyp_dict:
                hyp = set(hyp)
                prediction = set(prediction)
                score = len(hyp & prediction) /( len(hyp) + len(prediction))
                scores[-1].append(score)
            
        print (predictions[0], scores[0], true_scores[0])
        print (get_mean_true_score(np.array(true_scores), np.array(scores)))
        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

    if training_args.push_to_hub:
        model.save_to_hub("mimic-scorer")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    wandb.init()
    main()
