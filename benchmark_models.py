import numpy as np
import pandas as pd

import utils, analysis_utils

utils.init()

# Read models
with open("hyperparameters.pkl", "rb") as file:
    df = pd.read_pickle(file)

results = utils.tryLoadPickle("model_results.pkl", {})

for idx in range(df.index.max()+1):
    if idx in results:
        continue

    print(f"==========================\nModel: {idx}\n==========================")

    try:
        _, nearest = analysis_utils.benchmarkModel(df, idx)
        results[idx] = np.reshape(nearest, -1).tolist()
    except Exception as e:
        print(e)
        results[idx] = False
    
    utils.updatePickle("model_results.pkl", results)