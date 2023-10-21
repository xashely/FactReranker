import random
import numpy as np
import os
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from RadGraph import RadGraph
from nltk.tokenize import TreebankWordTokenizer as twt
from datasets import Dataset, DatasetDict, load_dataset, Value, ClassLabel, Features, Sequence
import pandas as pd

model_checkpoint = "StanfordAIMI/RadBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
RADGRAPH_LABELS = {"ANAT-DP": 0, "OBS-DP": 1, "OBS-U": 2, "OBS-DA": 3}
radgraph_scorer = RadGraph(reward_level="full", cuda=0, batch_size=1)


def read_hyps_refs(path):
    """Read hyps and refs from the file path

    Args:
        path: (str) the file path
    """
    print(f"Read hyps and refs from {path}")
    samples = []

    with open(f"{path}/openi_test_hyp_ref.json") as f:
        for line in f.readlines():
            samples.extend(json.loads(line))

    return samples


def build_dataset(raw_samples):
    index = 0
    samples = []
    sample_dict = {}


    def pre_graph(val):
        return {"tokens": val["tokens"], "label": val["label"], "relations": len(val["relations"]) != 0}

    temp_samples = []
    batch_size = 10
    start_index = 0
    # with open("/scratch/ace14856qn/scores.json", "r") as f:
    #     for line in f.readlines():
    #         start_index += 1 

    # print (f"Start processing from line {start_index}")

    for index in tqdm(range(start_index, len(raw_samples), batch_size)):
        temp_inputs = [[], [], []]
        for hyp, ori, ref in raw_samples[index:index+batch_size]:
            temp_inputs[0].append(hyp)
            temp_inputs[1].append(ori)
            temp_inputs[2].append(ref)
        _, true_scores, _, true_graphs = radgraph_scorer(refs=temp_inputs[2], hyps=temp_inputs[0], fill_cache=False)
        true_scores = true_scores[1]
        _, _, hyp_graphs, ref_graphs = radgraph_scorer(refs=temp_inputs[1], hyps=temp_inputs[0], fill_cache=False)
        ori = index
        for hyp, oris, score, hyp_graph, ref_graph, true_graph in zip(temp_inputs[0], temp_inputs[1], true_scores, hyp_graphs, ref_graphs, true_graphs):
            if index not in sample_dict:
                sample_dict[index] = {"ori": oris, "hyp": [], "score": [], "ref_graph": None, "hyp_graph": [], "true_graph": []}
        
            sample_dict[ori]["true_graph"] = [
                pre_graph(value) for value in true_graph["entities"].values()]
            sample_dict[ori]["ref_graph"] = [
                pre_graph(value) for value in ref_graph["entities"].values()]
            sample_dict[ori]["hyp"].append(hyp)
            sample_dict[ori]["hyp_graph"].append([
                pre_graph(value) for value in hyp_graph["entities"].values()
            ])
            sample_dict[ori]["score"].append(score)
            samples.append((hyp, oris, score, hyp_graph, ref_graph, true_graph))
            # with open("/scratch/ace14856qn/scores.json", "a") as f:
            #     f.write(json.dumps((hyp, ori, score, hyp_graph, ref_graph)))
            #     f.write("\n")
    index = 0
    overall_samples = []
    def decompose_val(value):
        temp_dict = {"tokens": [], "labels": [], "relations": []}
        for val in value:
            temp_dict["tokens"].append(val["tokens"])
            temp_dict["labels"].append(RADGRAPH_LABELS[val["label"]])
            temp_dict["relations"].append(val["relations"])
        return temp_dict
    hyp_index = 0
    for ref_index, (ref, value) in enumerate(sample_dict.items()):
        sample = {'query': value["ori"], 'positive': [], 'negative': [], 'hyp': [], 'score': [], 'hyp_graph_tokens': [], "hyp_graph_labels": [], "hyp_graph_relations": [], "ref_graph_tokens": [], "ref_graph_labels": [], "ref_graph_relations": []}
        ref_graph_dict = decompose_val(value["ref_graph"])
        true_graph_dict = decompose_val(value["true_graph"])
        sample["ref_graph_tokens"] = ref_graph_dict["tokens"]
        sample["ref_graph_labels"] = ref_graph_dict["labels"]
        sample["ref_graph_relations"] = ref_graph_dict["relations"]
        sample["true_graph_tokens"] = true_graph_dict["tokens"]
        sample["true_graph_labels"] = true_graph_dict["labels"]
        sample["true_graph_relations"] = true_graph_dict["relations"]
        sample["ref_index"] = ref_index
        max_score = max(value["score"])
        hyp_indexes = []
        for hyp, score in zip(value["hyp"], value["score"]):
            hyp_indexes.append(hyp_index)
            hyp_index += 1
            if abs(score - max_score) <= 1e-3:
                sample["positive"].append(hyp)
            else:
                sample["negative"].append(hyp)
        for graph in value["hyp_graph"]:
            graph_dict = decompose_val(graph)
            sample["hyp_graph_tokens"].append(graph_dict["tokens"])
            sample["hyp_graph_labels"].append(graph_dict["labels"])
            sample["hyp_graph_relations"].append(graph_dict["relations"])
        sample["hyp"] = value["hyp"]
        sample["hyp_index"] = hyp_indexes
        sample["score"] = value["score"]
        if len(sample["hyp"]) != 10:
            continue
        overall_samples.append(sample)
    # print(f"Overall samples: {len(overall_samples)}")
    # print(overall_samples[0])
    # ds = SentencesDataset(overall_samples, None)
    # dataloader = DataLoader(ds, batch_size=16)
    # return dataloader
    return from_samples_to_triple_dataset(overall_samples)

def from_samples_to_triple_dataset(samples):
    form_test_split = int(len(samples) * 0.8)
    train_validation_split = int(form_test_split * 0.8)
    indexes = list(range(len(samples)))
    # random.shuffle(indexes)
    # train_indexes = indexes[:train_validation_split]
    # valid_indexes = indexes[train_validation_split:form_test_split]
    # test_indexes = indexes[form_test_split:]
    samples = np.array(samples)
    # train_features = samples[train_indexes]
    # valid_features = samples[valid_indexes]
    # test_features = samples[test_indexes]

    def form_dataset(features):
        features = pd.DataFrame.from_records(features)
        print (features["score"].values.shape)
        scores = np.array([np.array(val) for val in features["score"].values])
        print (np.mean(np.amax(scores, axis=1)))
        print (np.mean(np.amin(scores, axis=1)))
        return Dataset.from_pandas(features, Features(
            {"query": Value("string"), "positive": Sequence(Value("string")), "negative": Sequence(Value("string")), "hyp": Sequence(Value("string")), "score": Sequence(Value("float32")),
                "ref_graph_tokens": Sequence(Value("string")),
                "ref_graph_labels": Sequence(Value("int64")),
                "ref_index": Value("int64"),
                "hyp_index": Sequence(Value("int64")),
                "ref_graph_relations": Sequence(Value("int64")),
                "true_graph_tokens": Sequence(Value("string")),
                "true_graph_labels": Sequence(Value("int64")),
                "true_graph_relations": Sequence(Value("int64")),
                "hyp_graph_tokens": Sequence(Sequence(Value("string"))),
                "hyp_graph_labels": Sequence(Sequence(Value("int64"))),
                "hyp_graph_relations": Sequence(Sequence(Value("int64"))),
            }))
    # train_dataset = form_dataset(train_features)
    # valid_dataset = form_dataset(valid_features)
    dataset = form_dataset(samples)
    # dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
    print (dataset)
    dataset.save_to_disk(f"/scratch/ace14856qn/scorer_true_openi_pair_dataset")
    return dataset

def from_samples_to_dataset(samples):
    form_test_split = int(len(samples) * 0.8)
    train_validation_split = int(form_test_split * 0.8)
    indexes = list(range(len(samples)))
    # random.shuffle(indexes)
    train_indexes = indexes[:train_validation_split]
    valid_indexes = indexes[train_validation_split:form_test_split]
    test_indexes = indexes[form_test_split:]
    samples = np.array(samples)
    train_features = samples[train_indexes]
    valid_features = samples[valid_indexes]
    test_features = samples[test_indexes]

    train_features = pd.DataFrame(train_features, columns=["sentence1", "sentence2", "label"])
    valid_features = pd.DataFrame(valid_features, columns=["sentence1", "sentence2", "label"])
    test_features = pd.DataFrame(test_features, columns=["sentence1", "sentence2", "label"])

    train_dataset = Dataset.from_pandas(train_features, Features({"sentence1": Value("string"), "sentence2": Value("string"), "label": Value("float32")}))
    valid_dataset = Dataset.from_pandas(valid_features, Features({"sentence1": Value("string"), "sentence2": Value("string"), "label": Value("float32")}))
    test_dataset = Dataset.from_pandas(test_features, Features({"sentence1": Value("string"), "sentence2": Value("string"), "label": Value("float32")}))

    dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
    dataset.save_to_disk(f"/scratch/ace14856qn/scorer_dataset")


if __name__ == "__main__":
    raw_contents = read_hyps_refs("/scratch/ace14856qn")
    build_dataset(raw_contents)
