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

def perceptron(data, labels, lr = 1):
  # Implement here
  np.random.seed(42)
  #w=np.random.rand(len(data[0]))
  w=np.ones(len(data[0]))
  batch_size=10
  num_iterations=1000
  #approach 1, works, sorta
  for _ in range(num_iterations):
      for sample_index,sample in enumerate(data):
        if(w.T @ sample <0):
            w = w + lr * sample
  '''
  this approach tried to first sum all missclassifications per iteration, didnt work
  for _ in range(num_iterations):
      missclass_samples=np.zeros(len(data[0]))
      for sample in data:
          if(w.T @ sample <0):
              missclass_samples+=sample
      w = w + lr *missclass_samples
      
  mini batch approach, got it closer to real but needs more fine tuning
  for itr in range(num_iterations):
    batch_indices=np.random.choice(len(data),size=batch_size, replace=False)
    batch=data[batch_indices]
    batch_labels=labels[batch_indices]
    missclass=[]
    for sample in batch:
        if(w.T @ sample<0):
            w=w+lr*sample
    '''
  return w

def plotAll(data,labels,w,bias):
    plt.scatter(data[labels == -1, 0], data[labels == -1, 1], label='Label -1', alpha=0.8)
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Label 1', alpha=0.7)
    a, b, c = w[0], w[1], bias
    m = -a / b
    b = -c / b

	# Generate some x values for the plot
    x = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.1)

	# Compute the corresponding y values using the equation of the line
    y= m * x +b
	# Plot the line
    plt.plot(x, y)

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Line")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

weights=perceptron(data,labels)
print(weights)

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
sk_p=Perceptron(tol=1e-3, random_state=42)
plotAll(orig_data,labels,weights[1:],weights[0])
sk_p.fit(orig_data,orig_labels)
print(f'sklearn perceptron weights: {sk_p.coef_[0]}\nMy weights: {weights[1:]}'
      f'\nsk perc bias:{sk_p.intercept_[0]}\n my bias:{weights[0]}')

