import pandas as pd
import numpy as np
import glob

for task in glob.glob("matbench_*"):
    try:
        d = pd.read_pickle(task + "/results/" + task + "_results" + ".pkl")
        print("{}: {}".format(task, np.mean(d["scores"])))
    except:
        pass
