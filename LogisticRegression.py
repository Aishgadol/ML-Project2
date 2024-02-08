import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D

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
    s