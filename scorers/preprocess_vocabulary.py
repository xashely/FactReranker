import json
from datasets import load_from_disk

raw_datasets = load_from_disk("/scratch/ace14856qn/scorer_process_openi_pair_dataset")
test_datasets = load_from_disk("/scratch/ace14856qn/scorer_true_openi_pair_dataset")


vocabulary = set()


for sample in raw_datasets:
    for word in sample["true_graph_tokens"]:
        vocabulary.add(word.lower())
    for word in sample["ref_graph_tokens"]:
        vocabulary.add(word.lower())
    for words in sample["hyp_graph_tokens"]:
        for word in words:
            vocabulary.add(word.lower())
for sample in test_datasets:
    for word in sample["true_graph_tokens"]:
        vocabulary.add(word.lower())
    for word in sample["ref_graph_tokens"]:
        vocabulary.add(word.lower())
    for words in sample["hyp_graph_tokens"]:
        for word in words:
            vocabulary.add(word.lower())


vocabulary = sorted(list(vocabulary))

with open("openi_entity_vocabulary", "w") as f:
    f.writelines(json.dumps(vocabulary))
print (len(vocabulary))

print (raw_datasets)
print (test_datasets)
