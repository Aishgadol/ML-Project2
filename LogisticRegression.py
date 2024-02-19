import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def plot(data, labels, w, bias):

  a, b, c = w[0], w[1], w[2]
  d = bias

  # create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='coolwarm')

  xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
  z = (-a * xx - b * yy - d) * 1.0 / c

  ax.plot_surface(xx, yy, z, alpha=0.4)
  ax.azim += 30
  ax.elev += 10
  #ax.view_init(elev=0, azim=90, roll=45)

  # customize the plot
  ax.set_xlim([0, np.max(data[:, 0])])
  ax.set_ylim([0, np.max(data[:, 1])])
  ax.set_zlim([0, np.max(data[:, 2])])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.title('3D Scatter Plot with 2D Labels')
  plt.show()

def plotRawData(data, labels):
      data_label_0 = [data[i] for i in range(len(data)) if labels[i] == -1]
      data_label_1 = [data[i] for i in range(len(data)) if labels[i] == 1]

      # Plotting
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      # Plot data points with label 0 (blue)
      ax.scatter(*zip(*data_label_0), c='g', label='Didnt purchase')

      # Plot data points with label 1 (red)
      ax.scatter(*zip(*data_label_1), c='r', label='Did purchase')

      ax.set_xlabel('Gender, 0 is male, 1 is female')
      ax.set_ylabel('Age')
      ax.set_zlabel('EstimatedSalary')

      plt.legend()
      plt.show()


def plotDensities(data):
    feature_names=df.columns.tolist()
    num_features = len(data[0])
    fig, axs = plt.subplots(1, num_features, figsize=(15, 4))

    for i in range(num_features):
        feature_values = [datapoint[i] for datapoint in data]
        axs[i].hist(feature_values, bins=20, density=True, color='red', alpha=0.8)
        axs[i].set_title(feature_names[i])
        axs[i].set_xlabel(feature_names[i])
        axs[i].set_ylabel('Density')

    plt.tight_layout()
    plt.show()
df = pd.read_csv('suv_data.csv')
df.dropna()
#dropping the label USER ID since it has no actual effect on our calculations
df=df.drop('User ID',axis=1)
#rewriting genders to 0/1 to use them for calculations
df['Gender']=df['Gender'].replace({'Male' : 0 , 'Female':1})
data=df.drop('Purchased',axis=1).to_numpy()
labels=df['Purchased'].to_numpy()
#switch all labels of 0 to -1 so we'll have labels in {-1,1}
labels[labels == 0] = -1
plotRawData(data,labels)
plotDensities(data)
#here we scale the data
mm_scaler=MinMaxScaler()
#added 1s tofirst row of every data sample to calculate bias term too
data=np.concatenate((np.ones((len(data),1)),data),axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train_scaled=np.concatenate((np.ones((len(X_train),1)),mm_scaler.fit_transform(X_train[:,1:])),axis=1)
X_test_scaled=np.concatenate((np.ones((len(X_test),1)),mm_scaler.transform(X_test[:,1:])),axis=1)
X_train_cutoff, X_val, y_train_cutoff, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
X_train_scaled_for_validation=np.concatenate((np.ones((len(X_train_cutoff),1)),mm_scaler.fit_transform(X_train_cutoff[:,1:])),axis=1)
X_val_scaled=np.concatenate((np.ones((len(X_val),1)),mm_scaler.transform(X_val[:,1:])),axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# For now, ignore the lambda, you will need it later
'''this version is unused because we found a simpler way, less spagetti
def Logistic_Regression_via_GD(P, y, lr, lamda=0):
    w=np.zeros(len(data[0]))
    num_iterations=20
    for _ in range(num_iterations):
        gradient=np.zeros(len(w))
        for x_index,x in enumerate(P):
            sigmoid_x=sigmoid(np.dot(x,w))
            if (sigmoid_x>0.5):
                if(y[x_index]==1):
                    gradient+=(-1*y[x_index]*x*(1-sigmoid_x))
                else:
                    gradient+=(-1*y[x_index]*x*sigmoid_x)
            elif (sigmoid_x<0.5):
                if(y[x_index]==-1):
                    gradient += (-1 * y[x_index] * x * (1 - sigmoid_x))
            else:
                gradient += (-1 * y[x_index] * x * sigmoid_x)
        w+=lr*gradient
    return w
'''
#this version is the one we're working with
def Logistic_Regression_via_GD(P,y,lr,lamda = 0):
    w=np.ones(P.shape[1])
    num_iterations=100
    for _ in range(num_iterations):
        gradient=np.zeros(len(w))
        for sample,label in zip(P,y):
            sigmoid_x=sigmoid(w.T @ sample)
            if(label==1):
                gradient+=(label*sample*(1-sigmoid_x))
            else:
                gradient+=(label*sample*sigmoid_x)
        w+=(lr*gradient + 2 * lamda * w)/len(P)
    return w[1:],w[0]

#predict function to predict output for single sample
def predict(x,w,b):
  sigmoid_pred=sigmoid(w.T @ x +b)
  if(sigmoid_pred>0.5):
    return 1
  return -1

def getBestLR_fromRange(givenRange):
    maxacc = 0
    bestlr = 0
    for lr in givenRange:
        w, b = Logistic_Regression_via_GD(X_train_scaled, y_train, lr)
        preds = np.zeros((len(y_train), 1))
        for sample_index, sample in enumerate(X_train_scaled):
            preds[sample_index] = predict(sample[1:], w, b)
        accuracy = sum(1 for pred, y_train_sample in zip(preds, y_train) if pred == y_train_sample) / len(y_train)
        if (accuracy > maxacc):
            maxacc = accuracy
            bestlr = lr
    return bestlr

bestfoundlr=getBestLR_fromRange(np.arange(0.1,10,0.1))
print(f'best lr found: {bestfoundlr}')
w, b = Logistic_Regression_via_GD(X_train_scaled, y_train, bestfoundlr)
preds = np.zeros((len(y_train), 1))
for sample_index, sample in enumerate(X_train_scaled):
    preds[sample_index] = predict(sample[1:], w, b)
accuracy = sum(1 for pred, y_train_sample in zip(preds, y_train) if pred == y_train_sample) / len(y_train)
print(f'accuracy: {accuracy}')

#now we run the model with test data (scaled tho), we'll print the accuracy and plot the hyperplane for the test data
#remember still no regulartization
w,b=Logistic_Regression_via_GD(X_train_scaled,y_train,bestfoundlr)
preds=np.zeros(len(y_test))
counts=0.0
for sample,label in zip(X_test_scaled,y_test):
    if(predict(sample[1:],w,b)==label):
        counts+=1
print(f'test accuracy is: {counts/len(y_test)}')
plot(X_test_scaled[:,1:], y_test, w, b)

'''To improve generalization, we use a tool that is called regularization.
In simple words,  Lloss(w)=Llogistic-reg(w)+λ⋅∥w∥^2 .
we updated the w,b calculating function and added + 2 * lamda * w to account for the regularization term'''