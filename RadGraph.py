import os
import torch.nn as nn
import numpy as np
import sys
import logging

from vilmedic.constants import EXTRA_CACHE_DIR
from vilmedic.zoo.utils import download_model
from vilmedic.blocks.scorers.RadGraph.utils import preprocess_reports
from utils_rad_graph import postprocess_reports

sys.path.append(os.path.join(os.path.dirname(__file__)))

logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("tqdm").setLevel(logging.CRITICAL)
logging.getLogger("filelock").setLevel(logging.CRITICAL)

from allennlp.commands.predict import _predict, _PredictManager
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.checks import check_for_gpu


class RadGraph(nn.Module):
    def __init__(
            self,
            lambda_e=0.5,
            lambda_r=0.5,
            batch_size=1,
            cuda=0,
            **kwargs
    ):

        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_r = lambda_r
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

    def forward(self, hyps):
        # Preprocessing
        number_of_reports = len(hyps)
        report_list = hyps
        model_input = preprocess_reports(report_list)
        # AllenNLP
        manager = _PredictManager(
            predictor=self.predictor,
            input_file=str(
                model_input
            ),  # trick the manager, make the list as string so it thinks its a filename
            output_file=None,
            batch_size=self.batch_size,
            print_to_console=False,
            has_dataset_reader=True,
        )
        results = manager.run()

        # Postprocessing
        inference_dict = postprocess_reports(results)
        # pickle.dump(inference_dict, open("./temp", "wb"))

        # Compute reward
        report_entity_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            report_entity = inference_dict[str(non_empty_report_index)]["entities"]
            report_entity_lists.append(report_entity)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_reports

        return report_entity_lists




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
