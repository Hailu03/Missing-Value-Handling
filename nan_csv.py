import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def min_max_normalization(X):
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    return X

def z_normalization(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X-mean)/std
    return X

# Read CSV file
iris = pd.read_csv('Iris.csv', delimiter=',', header=0, index_col=0)

# Convert categorical data to numeric data
iris['Species'] = pd.Categorical(iris.Species).codes

# Convert to numpy array
iris = np.array(iris)

# Shuffle data
randNum = np.arange(len(iris))
np.random.shuffle(randNum)
iris = iris[randNum]

# Divide data into train and test
X = iris[:, 0:4]
print("Please choose your choice to normalize data:")
print("A: No Normalization")
print("B: Min Max Normalization")
print("C: Standard Normalization")
choice = input("You: ")
if choice == "B":
    X = min_max_normalization(X)
elif choice == "C":
    X = z_normalization(X)
elif choice != "A" and choice != "B" and choice != "C":
    print("Invalid Input!")

y = iris[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# non nan iris
non_nan_iris = pd.DataFrame(X_train, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
non_nan_iris['Species'] = y_train
non_nan_iris.to_csv(f'Iris_Filling_Target.csv', index=False)

# Calculate % of the total number of values in the dataset
prob = 0.05
for i in range(4):
    nan_mask = np.random.choice([False, True], size=X_train.shape, p=[1-prob, prob])
    # Replace of data points with NaN values
    temp = X_train.copy()
    temp[nan_mask] = np.nan

    # Save the dataset with NaN values to a new CSV file
    temp = pd.DataFrame(temp, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    temp['Species'] = y_train
    temp.to_csv(f'IrisNan{int(prob * 100)}.csv', index=False)
    prob += 0.05
