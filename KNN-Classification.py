import matplotlib as mb
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def safe_read(path):
    return pd.read_csv(
        path,
        sep=r"\s+",          # split on any whitespace
        header=None,         
        engine="python",     # more forgiving parser
        on_bad_lines="skip"  # skip malformed lines
    )

testDataOne   = safe_read(r"Data/classification/TestData1.txt")
testDataTwo   = safe_read(r"Data/classification/TestData2.txt")
testDataThree = safe_read(r"Data/classification/TestData3.txt")
testDataFour  = safe_read(r"Data/classification/TestData4.txt")