import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import requests
from io import BytesIO

def load_npy_file(url):
  response = requests.get(url)
  if response.status_code == 200:
    npy_data = np.load(BytesIO(response.content), allow_pickle=True).item()
    return npy_data
  else:
    return None

data_dict=load_npy_file('https://sharon.srworkspace.com/ml/datasets/hw2/svm_data_2d.npy')
X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']
def plotData():
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Train')
    plt.show()

def phi_func(x):
    return np.concatenate((np.square(x),x),axis=1)

new_features=phi_func(X_train)
model = SVC(kernel='linear', C=10)
model.fit(new_features, y_train)

# Get the hyperplane equation coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_

# Print the hyperplane equation
equation_parts = []
for i in range(len(coefficients)):
    equation_parts.append(f"({coefficients[i]:.3f} * X{i+1})")
equation = " + ".join(equation_parts) + f" + ({intercept[0]:.3f})"

print("Hyperplane equation:")
print(f"  {equation}")
train_features = phi_func(X_train)
train_preds = model.predict(train_features)
train_acc = np.sum(y_train == train_preds) / len(y_train)

val_features = phi_func(X_val)
val_preds = model.predict(val_features)
val_acc = np.sum(y_val == val_preds) / len(y_val)

def plotLinearSVM():
    xx, yy = np.meshgrid(np.arange(-2, 2.2, 0.1), np.arange(-2, 2.2, 0.1))
    data = np.c_[xx.ravel(), yy.ravel()]

    new_features = phi_func(np.concatenate((X_train, X_val)))
    Z = model.predict(phi_func(data))
    Z = Z.reshape(xx.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the training data on the first subplot
    axs[0].contourf(xx, yy, Z, alpha=0.8)
    scatter1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(f'Train dataset - {train_acc:.4f} accuracy')

    # Plot the validation data on the second subplot
    axs[1].contourf(xx, yy, Z, alpha=0.8)
    scatter2 = axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='bwr')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(f'Validation dataset - {val_acc:.4f} accuracy')

    plt.show()

def plotPolySVM():
    model = SVC(kernel='poly', degree=4, C=10)
    clf = model.fit(X_train, y_train)

    xx, yy = np.meshgrid(np.arange(-2, 2.2, 0.1), np.arange(-2, 2.2, 0.1))
    xy = np.c_[xx.ravel(), yy.ravel()]

    P = model.decision_function(xy).reshape(xx.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the training data on the first subplot
    axs[0].contourf(xx, yy, P, alpha=0.8)
    scatter1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(f'Train dataset - {clf.score(X_train, y_train):.4f} accuracy')

    # Plot the validation data on the second subplot
    axs[1].contourf(xx, yy, P, alpha=0.8)
    scatter2 = axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='bwr')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(f'Validation dataset - {clf.score(X_val, y_val):.4f} accuracy')

    plt.show()
def myKernel(x,y):
    return np.dot(phi_func(x),phi_func(y).T)
def plotPhi():
    model = SVC(kernel=myKernel, gamma=1.5, C=0.1)
    clf = model.fit(X_train, y_train)
    xx, yy = np.meshgrid(np.arange(-2, 2.2, 0.1), np.arange(-2, 2.2, 0.1))
    xy = np.c_[xx.ravel(), yy.ravel()]
    P = model.decision_function(xy).reshape(xx.shape)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Plot the training data on the first subplot
    axs[0].contourf(xx, yy, P, alpha=0.8)
    axs[0].contour(xx, yy, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'],linewidths=2)
    scatter1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(f'Train dataset - {clf.score(X_train, y_train):.4f} accuracy')

    # Plot the validation data on the second subplot
    axs[1].contourf(xx, yy, P, alpha=0.8)
    axs[1].contour(xx, yy, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'],linewidths=2)
    scatter2 = axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='bwr')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(f'Validation dataset - {clf.score(X_val, y_val):.4f} accuracy')

    plt.show()
plotPhi()
def plotRBF_SVM():
    model = SVC(kernel='rbf', gamma=1.5, C=10)
    clf = model.fit(X_train, y_train)

    xx, yy = np.meshgrid(np.arange(-2, 2.2, 0.1), np.arange(-2, 2.2, 0.1))
    xy = np.c_[xx.ravel(), yy.ravel()]

    P = model.decision_function(xy).reshape(xx.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the training data on the first subplot
    axs[0].contourf(xx, yy, P, alpha=0.8)
    scatter1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(f'Train dataset - {clf.score(X_train, y_train):.4f} accuracy')

    # Plot the validation data on the second subplot
    axs[1].contourf(xx, yy, P, alpha=0.8)
    scatter2 = axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='bwr')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(f'Validation dataset - {clf.score(X_val, y_val):.4f} accuracy')

    plt.show()


def findBest_acc_gamma_c(gammas,cees):
    gammas = gammas
    cees = cees
    accs = []
    max_acc = 0
    best_gamma = 0
    best_c = 0
    for gamma in gammas:
        for c in cees:
            model = SVC(kernel='rbf', gamma=gamma, C=c)
            clf = model.fit(X_val, y_val)
            accs.append(clf.score(X_val, y_val))
            if (clf.score(X_val, y_val) > max_acc):
                print(f'found new best, gamma:{best_gamma}, c:{best_c}, with acc={max_acc}')
                best_gamma = gamma
                max_acc=clf.score(X_val, y_val)
                best_c = c
    return accs,max_acc,best_gamma,best_c

gammas = np.arange(0,20,0.5)
cees = range(1,30)
accs,max_acc,best_gamma,best_c=findBest_acc_gamma_c(gammas,cees)
plt.figure(figsize=(15,6))
combinations = [f"{a}, {b}" for a in gammas for b in cees]
plt.plot(combinations, accs, color='red')
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.title(f'Tuning, Best gamma: {best_gamma}, best c: {best_c}, max acc for them: {max_acc}')
plt.xticks(combinations)
plt.show()
