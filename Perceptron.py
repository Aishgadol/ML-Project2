import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

column_names = ['Feature1', 'Feature2', 'Label']
df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams.csv',header=None,names=column_names)
#print(df)
#seperate data to features and labels
data=df.drop('Label',axis=1).to_numpy()
labels=df['Label'].to_numpy()
#set all labels 0 to -1 to work with perceptron algorithm
labels[labels==0]=-1
