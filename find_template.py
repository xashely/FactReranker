import json
import evaluate
import numpy as np


template_file = "/scratch/ace14856qn/tg_result.json"


def read_json(filename):
    with open(filename) as f:
        a = f.readlines()
    data = []
    for val in a:
        try:
           load_val = json.loads(val)
           data.append(load_val)
        except Exception as e:
           val = val.split('{"raw_data')
           load_val_pre = json.loads('{"raw_data'+val[1])
           load_val_after = json.loads('{"raw_data'+val[2])
           data.append(load_val_pre)
           data.append(load_val_after)
    return data


def best_over_one(data):
    for datum in data:
        best_score = datum["raw_data"]["score"][np.argmax(datum["estimate_score"])]
        datum["best_over_one"] = best_score - datum["raw_data"]["score"][0]

data = read_json(template_file)
best_over_one(data)
# data = sorted(data, key=lambda x:x["best_over_one"])
print ("########")
# for datum in data[-10:]:
#     print (datum["raw_data"]["score"], datum["estimate_score"])
#     print ("------")
selected_datum = [val for val in data if "focal airspace consolidation" in val["raw_data"]["query"]][0]
print (selected_datum)


entity_labels = ["[ANAT-DP]", "[OBS-DP]", "[OBS-U]", "[OBS-DA]", "[NO]"]
relation_labels = ["[NA]", "[REL]", "[FAIL]"]

def build_prediction_strings(predictions, evaluate=False):
    predictions = [(val[0], entity_labels[val[1]], relation_labels[val[2]]) for val in predictions]
    predictions = [" ".join(val) for val in predictions]
    predictions = " [ENT] ".join(predictions)
    if evaluate:
        predictions = "<s>"+predictions+"</s>"
    return predictions

# pred = build_prediction_strings(selected_datum["predictions"])
# print (pred)

def build_replaced_strings(raw_string, tokens, labels, relations):
    print (raw_string)
    labels = [entity_labels[val][1:-1] for val in labels]
    relations = [relation_labels[val][1:-1] for val in relations]
    graph_dict = list(zip(tokens, labels, relations))
    # graph_dict = [("\\textcolor{red}{("+v[0], v[1], v[2]+")}") for v in graph_dict]
    graph_dict = [("("+v[0], v[1], v[2]+")") for v in graph_dict]
    graph_dict = [", ".join(val) for val in graph_dict]
    raw = raw_string.split()
    new_raw = []
    visiting_index = 0
    print (graph_dict)
    for (index, r) in enumerate(raw):
        if r == "in" and raw[index+1] == "place.":
            r = "in place"
        if r == "place." and raw[index-1] == "in":
            continue
        r = r.strip(".").strip(",")
        if r.strip(".").strip(",") not in tokens[visiting_index:]:
            new_raw.append(r)
        else:
            print (r, r in tokens[visiting_index:], visiting_index, tokens[visiting_index])
            new_raw.append(graph_dict[visiting_index])
            visiting_index += 1
    return " ".join(new_raw)


ref_string = build_replaced_strings(selected_datum["raw_data"]["query"], selected_datum["raw_data"]["ref_graph_tokens"], selected_datum["raw_data"]["ref_graph_labels"], selected_datum["raw_data"]["ref_graph_relations"])
print (ref_string)
# true_string = build_replaced_strings("1.  Hazy opacity in the right lung which may represent aspiration versus\n pleural effusion or hemorrhage.\n \n 2.  Mild pulmonary edema.\n \n 3.  No displaced rib fractures.", selected_datum["raw_data"]["true_graph_tokens"], selected_datum["raw_data"]["true_graph_labels"], selected_datum["raw_data"]["true_graph_relations"])

# print (true_string)


metric = evaluate.load("rouge")
metric_bert = evaluate.load("bertscore")

true_strings = []
generate_strings = []
rad_scores = []

def evaluate_datum(datum):
    graph_dict = list(zip(datum["raw_data"]["true_graph_tokens"], datum["raw_data"]["true_graph_labels"], datum["raw_data"]["true_graph_relations"]))
    ref_graph_dict = list(zip(datum["raw_data"]["ref_graph_tokens"], datum["raw_data"]["ref_graph_labels"], datum["raw_data"]["ref_graph_relations"]))
    true_string = build_prediction_strings(graph_dict)
    true_strings.append(true_string)
    print (true_strings)
    print (build_prediction_strings(ref_graph_dict))
    generate_string = build_prediction_strings(datum["predictions"])
    generate_strings.append(generate_string)
    print (generate_strings)
    prediction_set = [" ".join([str(v) for v in val]) for val in datum["predictions"]]
    label_set = [" ".join([str(v) for v in val]) for val in graph_dict]
    # print (prediction_set)
    # print (label_set)
    # print (len(set(prediction_set) & set(label_set)), len(set(prediction_set)), len(set(label_set)))
    if prediction_set or label_set:
        rad_score = (2 * len(set(prediction_set) & set(label_set))) / (len(set(prediction_set))+len(set(label_set)))
    else:
        rad_score = 0.0
    rad_scores.append(rad_score) 

evaluate_datum(selected_datum)
# for datum in data:
#     evaluate_datum(datum)
# result = metric.compute(predictions=generate_strings, references=true_strings, use_stemmer=True)
# print (result)
# result = metric_bert.compute(predictions=generate_strings, references=true_strings, lang="en")
# result = {"bertscore_f1":round(v, 4) for v in result["f1"]}
# print (result)
# print (np.mean(rad_scores))

    
        


