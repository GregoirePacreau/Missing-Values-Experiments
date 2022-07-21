# This file allows to communicate with the R packages implementing robust statistics

import subprocess
import pandas as pd
import numpy as np
import os

def DDC(data):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DDC_data.csv")

    # Calls the script which produces a boolean matrix detecting ouliers using DDC
    os.system("Rscript R_scripts/DDC.R temp/DDC_data.csv")

    res = pd.read_csv("temp/DDC_data_res.csv")
    os.remove("temp/DDC_data.csv")
    os.remove("temp/DDC_data_res.csv")
    return res

def TSGS(data, filter="UBF-DDC", partial_impute=False, tol=1e-4, maxiter=150, method="bisquare", init="emve"):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/TSGS_data.csv")

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/TSGS.R temp/TSGS_data.csv {} {} {} {} {}".format(
        str(filter), str(partial_impute), str(tol), str(maxiter), method, init
    ))

    res = pd.read_csv("temp/TSGS_data_res.csv")
    os.remove("temp/TSGS_data.csv")
    os.remove("temp/TSGS_data_res.csv")
    return res

def DI(data, initEst="DDCWcov", crit=0.01, maxits=10, quant=0.99, maxCol=0.25):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DI_data.csv")

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/DI.R temp/DI_data.csv {} {} {} {} {}".format(
        str(initEst), str(crit), str(maxits), str(quant), str(maxCol)
    ))

    res = pd.read_csv("temp/DI_data_res.csv")
    os.remove("temp/DI_data.csv")
    os.remove("temp/DI_data_res.csv")
    return res

def DDCwcov(data, maxCol=0.25):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DDCwcov_data.csv")

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/DDCwcov.R temp/DDCwcov_data.csv " + str(maxCol))

    res = pd.read_csv("temp/DDCwcov_data_res.csv")
    os.remove("temp/DDCwcov_data.csv")
    os.remove("temp/DDCwcov_data_res.csv")
    return res

if __name__ == "__main__":
    data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=50)
    res = DDC(data)
    print(res)