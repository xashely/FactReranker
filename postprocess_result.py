import json
import numpy as np
import nltk
import evaluate
from RadGraph import RadGraph
from vilmedic.blocks.scorers.CheXbert.chexbert import CheXbert
from vilmedic.blocks.scorers.RadEntityMatchExact.RadEntityMatchExact import RadEntityMatchExact
from vilmedic.blocks.scorers.RadEntityNLI.RadEntityNLI import RadEntityNLI

with open("chatgpt_result_openi.json") as f:
    data = f.readlines()

data = [json.loads(val) for val in data]


def process_prediction(pred):
    keyword = "IMPRESSION:"
    if keyword in pred:
        return pred[pred.index(keyword) + len(keyword):].strip()
    else:
        return pred.strip()

processed_preds = [process_prediction(val["pred"]) for val in data]
processed_labels = [" ".join(val["label"]) for val in data]

print (processed_preds[:5], processed_labels[:5])
def postprocess(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

metric = evaluate.load("rouge")
metric_bert = evaluate.load("bertscore")
decoded_preds, decoded_labels = postprocess(processed_preds, processed_labels)
result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
result_bert = metric_bert.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
result_radentity = {}
rad_ref_tokens = [" ".join(val.split()) for val in decoded_labels]
rad_hyp_tokens = [" ".join(val.split()) for val in decoded_preds]
result_radentity["radgraph_simple"], result_radentity["radgraph_partial"], result_radentity["radgraph_complete"] = \
	RadGraph(reward_level="full")(refs=rad_ref_tokens, hyps=rad_hyp_tokens)[0]

result_chexbert = {}
accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = CheXbert()(hyps=decoded_preds, refs=decoded_labels)
result_chexbert["chexbert_five_f1"] = chexbert_5["micro avg"]["f1-score"]
result_chexbert["chexbert_all_f1"] = chexbert_all["micro avg"]["f1-score"]
result_chexbert["chexbert_all_precision"] = chexbert_all["micro avg"]["precision"]
result_chexbert["chexbert_all_recall"] = chexbert_all["micro avg"]["recall"]
result = {k: round(v * 100, 4) for k, v in result.items()}
result_bert = {"bertscore_f1":round(v, 4) for v in result_bert["f1"]}

# result["gen_len"] = np.mean(prediction_lens)
result.update(result_bert)
result.update(result_chexbert)
result.update(result_radentity)
print (result)
