import numpy as np
import pandas as pd

import utils, analysis_utils

utils.init()

# Read models
with open("hyperparameters.pkl", "rb") as file:
    df = pd.read_pickle(file)

results = utils.tryLoadPickle("model_results.pkl", {})
jobs = []
for idx in range(df.index.max()+1):
    if not idx in results:
        jobs.append(idx)

def work(idx):
    print(f"==========================\nModel: {idx}\n==========================")
    try:
        _, nearest = analysis_utils.benchmarkModel(df, idx)
        return idx, np.reshape(nearest, -1).tolist()
    except Exception as e:
        print(e)
        return idx, False
    
def result(result):
    k, v = result
    results = utils.tryLoadPickle("model_results.pkl", {})
    results[k] = v
    utils.updatePickle("model_results.pkl", results)

analysis_utils.MultithreadedJob(jobs, work, result).run()
print("Finished!")