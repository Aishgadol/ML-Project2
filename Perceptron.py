import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sklearn.preprocessing import MinMaxScaler

column_names = ['Feature1', 'Feature2', 'Label']
df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams.csv',header=None,names=column_names)
#keep the original data somewhere, we'll scale it later for easier use
orig_data=df.drop('Label',axis=1).to_numpy()
orig_labels=df['Label'].to_numpy()
# #seperate data to features and labels
data=df.drop('Label',axis=1).to_numpy()
labels=df['Label'].to_numpy()
#set all labels 0 to -1 to work with perceptron algorithm
labels[labels==0]=-1
def plotPoints(data):
    plt.scatter(data[labels == -1, 0], data[labels == -1, 1], label='Label -1', alpha=0.8)
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Label 1', alpha=0.7)

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Scatter Plot of Features with Labels')

    # Show legend
    plt.legend()
    # Show the plot
    plt.show()

#plotPoints(data)

#create scaler
mm_scaler=MinMaxScaler()
#fit the data and transform it according to scaler
orig_data=mm_scaler.fit_transform(orig_data)
data=mm_scaler.fit_transform(data)
#add 1's in dimension 0 to take bias into consideration
data = np.insert(data, 0, 1, axis=1)
#multiply every data that has label -1 with -1 so that for all x in dataset, g_i(x)>0
data[labels==-1]*=(-1)
