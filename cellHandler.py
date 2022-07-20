# This file allows to communicate with the R packages implementing robust statistics

import subprocess
import pandas as pd

def DDC(data):
    # Data is a pandas Dataframe
    if not data.isinstance(pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("DDC_data.csv")
    subprocess.subprocess.run(['Rscript', 'DDC.R', "DDC_data.csv"])

    res = pd.read_csv("DDC_data_res.csv")