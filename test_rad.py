from RadGraph import RadGraph
import json
radgraph_scorer = RadGraph(reward_level="full", batch_size=1)

with open("test.log") as f:
    a = f.readlines()[-1]

a = json.loads(a)
batch_size = 10
    
for start_index in range(0, len(a), batch_size):
    text = a[start_index:start_index+batch_size]
    text = [" ".join(val.split()[:230]) for val in text]
    print (start_index, [len(val.split()) for val in text])
    radgraph_scorer(refs=text, hyps=text, fill_cache=False)
