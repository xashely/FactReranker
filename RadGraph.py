import os
import torch.nn as nn
import numpy as np
import sys
import logging
import json
import re
import traceback

from vilmedic.constants import EXTRA_CACHE_DIR
from vilmedic.zoo.utils import download_model
from vilmedic.blocks.scorers.RadGraph.utils import (
    get_entity,
    preprocess_reports,
    postprocess_reports,
    compute_reward,
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
#
logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("tqdm").setLevel(logging.CRITICAL)
logging.getLogger("filelock").setLevel(logging.CRITICAL)

from allennlp.commands.predict import _predict, _PredictManager
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.checks import check_for_gpu


def preprocess_reports(report_list):
    """Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    final_list = []
    for idx, report in enumerate(report_list):
        sen = re.sub(
            "(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )", r" ", report
        ).split()
        temp_dict = {}

        temp_dict["doc_key"] = str(idx)

        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

    final_lengths = [len(val["sentences"][0]) for val in final_list]
    doc_lengths = [len(val["sentences"]) for val in final_list]
    final_list = [v for val in final_list for v in val["sentences"]]
    batch_size = 10
    final_lists = []
    for start_index in range(0, len(final_list), batch_size):
        final_lists.append({"doc_key": str(start_index), "sentences": final_list[start_index:start_index+batch_size]}) 
    # final_list = [{"doc_key": "0", "sentences": final_list}]

    return [json.dumps(item) for item in final_lists], final_lengths


def postprocess_reports(results, lengths):
    """Post processes all the reports and saves the result in train.json format"""
    final_dict = {}
    data = []
    # overall_data = {"sentences": [], "predicted_ner": [], "predicted_relations": []}
    doc_index = 0
    sen_index = 0
    cum_length = 0

    for r in results:
        datum = json.loads(r)
        for sentence, predicted_ner, predicted_relation in zip(
            datum["sentences"],
            datum["predicted_ner"],
            datum["predicted_relations"],
        ):
            data.append({"sentences": [sentence], "predicted_ner": [predicted_ner], "predicted_relations": [predicted_relation], "doc_key": str(doc_index)})
            doc_index += 1
        # data.append(json.loads(r))
        # overall_data["sentences"].extend(data[-1]["sentences"])
        # overall_data["predicted_ner"].extend(data[-1]["predicted_ner"])
        # overall_data["predicted_relations"].extend(data[-1]["predicted_relations"])

    # data = [overall_data]
    pre_length = 0

    for file, length in zip(data, lengths):
        if int(file["doc_key"]) % 10 == 0:
            pre_length = 0
        doc_dict = postprocess_individual_report(file, lengths=pre_length)
        pre_length += length
        final_dict[file["doc_key"]] = doc_dict

    return final_dict


def postprocess_individual_report(file, data_source=None, lengths=None):
    """Postprocesses individual report
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    temp_dict = {}
    temp_dict["text"] = " ".join(file["sentences"][0])
    n = file["predicted_ner"][0]
    n = [[val[0] - lengths, val[1] - lengths, val[2], val[3], val[4]] for val in n]
    r = file["predicted_relations"][0]
    r = [[val[0] - lengths, val[1] - lengths, val[2] - lengths, val[3] - lengths, val[4], val[5], val[6]] for val in r]
    s = file["sentences"][0]
    temp_dict["entities"] = get_entity(n, r, s)
    temp_dict["data_source"] = data_source
    temp_dict["data_split"] = "inference"

    # final_dict[file["doc_key"]] = temp_dict
    return temp_dict

    pre_length = 0
    if "predicted_ner" not in file:
        print([len(val) for val in file["sentences"]])
    for index, (sentence, n, r, l) in enumerate(zip(file["sentences"], file["predicted_ner"], file["predicted_relations"], lengths)):
        try:
            temp_dict = {}

            temp_dict["text"] = " ".join(sentence)
            # n = [[val[0] - pre_length, val[1] - pre_length, val[2], val[3], val[4]] for val in n]
            # pre_length += l
            temp_dict["entities"] = get_entity(n, r, sentence)
            temp_dict["data_source"] = data_source
            temp_dict["data_split"] = "inference"

            final_dict[str(index)] = temp_dict

        except Exception:
            traceback.print_exc()
            print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")


class RadGraph(nn.Module):
    def __init__(
            self,
            lambda_e=0.5,
            lambda_r=0.5,
            reward_level="full",
            batch_size=1,
            cuda=0,
            **kwargs
    ):

        super().__init__()
        assert reward_level in ["simple", "complete", "partial", "full"]
        self.lambda_e = lambda_e
        self.lambda_r = lambda_r
        self.reward_level = reward_level
        self.cuda = cuda
        self.batch_size = batch_size

        self.model_path = os.path.join(EXTRA_CACHE_DIR, "radgraph.tar.gz")

        if not os.path.exists(self.model_path):
            download_model(
                repo_id="StanfordAIMI/RRG_scorers",
                cache_dir=EXTRA_CACHE_DIR,
                filename="radgraph.tar.gz",
            )

        # Model
        import_plugins()
        import_module_and_submodules("dygie")

        check_for_gpu(self.cuda)
        archive = load_archive(
            self.model_path,
            weights_file=None,
            cuda_device=self.cuda,
            overrides="",
        )
        self.predictor = Predictor.from_archive(
            archive, predictor_name="dygie", dataset_reader_to_load="validation"
        )
        self.manager = _PredictManager(
                predictor=self.predictor,
                input_file="",
                output_file=None,
                batch_size=1,
                print_to_console=False,
                has_dataset_reader=True,
            )

        # with open("/scratch/ace14856qn/cache.json") as f:
        #     self.cache = json.loads(f.readlines()[0])
        self.cache = {}

    def forward(self, refs, hyps, fill_cache=True):
        # Preprocessing
        number_of_reports = len(hyps)

        assert len(refs) == len(hyps)

        empty_report_index_list = [
            i
            for i in range(number_of_reports)
            if (len(hyps[i]) == 0) or (len(refs[i]) == 0)
        ]
        cached_hyps_index_list = [
            i
            for i in range(number_of_reports)
            if hyps[i] in self.cache or i in empty_report_index_list
        ]
        cached_refs_index_list = [
            i
            for i in range(number_of_reports)
            if refs[i] in self.cache or i in empty_report_index_list
        ]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)
        number_of_non_cached_hyps = number_of_reports - len(cached_hyps_index_list)
        number_of_non_cached_refs = number_of_reports - len(cached_refs_index_list)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list and i not in cached_hyps_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list and i not in cached_refs_index_list
                      ]
        print (len(hyps), len(refs))
        # print (f"Cache hit ratio {len(report_list) / (len(hyps) + len(refs))}")

        # assert len(report_list) == 2 * number_of_non_empty_reports

        # import pickle
        # if os.path.exists("./temp"):
        #     inference_dict = pickle.load(open("./temp", "rb"))
        # else:
        if report_list:
            model_input, lengths = preprocess_reports(report_list)
            all_results = []
            batch_size = 10
            for start_index in range(0, len(model_input), batch_size):
                self.manager._input_file = str(model_input[start_index:start_index+batch_size])
                results = self.manager.run()
                all_results.extend(results)
            results = all_results

        # Postprocessing
            inference_dict = postprocess_reports(results, lengths)
        # pickle.dump(inference_dict, open("./temp", "wb"))
        # print (len(inference_dict.keys()))

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        hyps_index = 0
        refs_index = number_of_non_cached_hyps
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                if self.reward_level == "full":
                    reward_list.append((0., 0., 0.))
                else:
                    reward_list.append(0.)

                continue

            if report_index not in cached_hyps_index_list:
                hypothesis_annotation_list = inference_dict[str(hyps_index)]
                self.cache[hyps[report_index]] = hypothesis_annotation_list
                hyps_index += 1
            else:
                hypothesis_annotation_list = self.cache[hyps[report_index]]

            if report_index not in cached_refs_index_list:
                reference_annotation_list = inference_dict[str(refs_index)]
                self.cache[refs[report_index]] = reference_annotation_list
                refs_index += 1
            else:
                reference_annotation_list = self.cache[refs[report_index]]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.lambda_e,
                    self.lambda_r,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports
        if fill_cache:
            with open("/scratch/ace14856qn/cache.json", "w") as f:
                f.write(json.dumps(self.cache))

        if self.reward_level == "full":
            reward_list_ = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            reward_list = reward_list_
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward = np.mean(reward_list)

        # print (reward_list)

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


if __name__ == "__main__":
    import time

    m = RadGraph(cuda=0, reward_level="partial", batch_size=1)
    # report = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    # hypothesis_report_list = [report, "", "a", report]
    #
    # report_2 = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The heart is clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    # reference_report_list = [report_2, report_2, report_2, report_2]
    #
    # reward_list = m(hyps=hypothesis_report_list, refs=reference_report_list)
    t = time.time()
    num = str(103276)
    l1 = open("test_best-1_881942_hyps.txt").readlines()
    # l1 = [l.strip() for l in l1][:10]
    l1 = [l.strip() for l in l1]
    l2 = open("test_best-1_103276_refs.txt").readlines()
    # l2 = [l.strip() for l in l2][:10]
    l2 = [l.strip() for l in l2]
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = m(hyps=l1, refs=l2)
    # print(time.time() - t)
    print(mean_reward)  # [0.8666666666666667, 0, 0, 0.8666666666666667]


# ^[(0.353946348023485, 0.32697070866071776, 0.25986992412367665)
