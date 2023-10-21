import logging
import numpy as np
import os
import csv
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import wandb

logger = logging.getLogger(__name__)

class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, mrr_at_k: int = 10, name: str = '', write_csv: bool = True):
        def preprocess_dataset(dataset):
            lengths = []
            labels = []
            examples = []
            for example in dataset:
                is_relevant = []
                for hyp, hyp_graph_tokens, hyp_graph_labels, hyp_graph_relations, score in zip(example["hyp"], example["hyp_graph_tokens"], example["hyp_graph_labels"], example["hyp_graph_relations"], example["score"]):
                    examples.append({"hyp": hyp, "ref": example["query"], "label": score,
                        "hyp_graph_tokens": hyp_graph_tokens, 
                        "hyp_graph_labels": hyp_graph_labels, 
                        "hyp_graph_relations": hyp_graph_relations, 
                        "ref_graph_tokens": example["ref_graph_tokens"], 
                        "ref_graph_labels": example["ref_graph_labels"], 
                        "ref_graph_relations": example["ref_graph_relations"], 
                    })
                    is_relevant.append(hyp in example["positive"])
                lengths.append(len(example["hyp"]))
                labels.append(is_relevant)
            ds = SentencesDataset(examples, None)
            dataloader = DataLoader(ds, batch_size=1024)
            return dataloader, labels, lengths
        self.dataloader, self.labels, self.lengths = preprocess_dataset(samples)
        self.name = name
        self.mrr_at_k = mrr_at_k
        sample_dict = {}
        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MRR@{}".format(mrr_at_k)]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, all_loss: float = 1.0) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        pred_scores = model.predict(self.dataloader, convert_to_numpy=True, show_progress_bar=True).reshape(-1)
        preds = []
        start_index = 0
        for length in self.lengths:
            preds.append(pred_scores[start_index:start_index+length])
            start_index += length
        for score, label in tqdm(zip(preds, self.labels)):
            num_queries += 1
            num_positives.append(sum(label))
            num_negatives.append(len(label) - sum(label))

            # ote = model.encode(query, convert_to_tensor=True)
            # te = torch.stack([model.encode(doc, convert_to_tensor=True) for doc in docs])
            # pred_scores = cosine_similarity(ote, te).cpu()
            pred_scores_argsort = np.argsort(-score)  #Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if label[index]:
                    mrr_score = 1 / (rank+1)
                    break

            all_mrr_scores.append(mrr_score)

        mean_mrr = np.mean(all_mrr_scores)
        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        wandb.log({
            "num": {
                "queries": num_queries,
                "positive_min": np.min(num_positives),
                "positive_mean": np.mean(num_positives),
                "positive_max": np.max(num_positives),
                "negative_min": np.min(num_negatives),
                "negative_mean": np.mean(num_negatives),
                "negative_max": np.max(num_negatives),
            },
            f"mrr@{self.mrr_at_k}": mean_mrr*100,
            f"train_loss": all_loss,
        })
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr])

        return mean_mrr
