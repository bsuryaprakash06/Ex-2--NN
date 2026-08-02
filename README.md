| **Name** | **Register Number** | **Experiment No.** | **Date** |
|-----------|---------------------|--------------------|----------|
| Surya Prakash B | 212224230281 | 2 | 31 - 07 - 2026 |

## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# Perceptron – Theoretical Concept

A **Perceptron** is a basic learning algorithm invented in **1959** by **Frank Rosenblatt**. It is designed to mimic the working of a biological neuron.

The human brain consists of many interconnected neurons. Each neuron:
- Receives a set of inputs.
- Performs some computation.
- Passes the result to other neurons.

Similarly, a perceptron receives input features, performs a computation, and produces an output.

A **Perceptron** is a supervised learning algorithm used for **binary classification**. It classifies data into one of two classes by assigning weights to the input features.

---

# Working of a Perceptron

A perceptron works in two phases:

## 1. Training Phase

- Weights are initialized with arbitrary values.
- The perceptron evaluates a training sample.
- The predicted output is compared with the actual output.
- If the prediction is incorrect, the weights and bias are adjusted.
- This process is repeated until the model learns the correct decision boundary.

## 2. Testing Phase

- After training, the perceptron is tested using new and unseen samples.
- The testing phase checks whether the trained model can correctly classify real-world data.

---

# Important Points

- A perceptron requires a large number of already classified (labeled) training samples.
- The required number of samples depends on the complexity of the problem.
- Real-world data is rarely perfect, so data preprocessing is required.
- Preprocessing includes:
  - Handling missing values
  - Removing unrelated data
  - Feature scaling
- A perceptron works only for **linearly separable** data.

---

# Mathematical Model

If each sample has **n** features, the perceptron computes a weighted sum of the inputs.

**Output Function**

```
f(x) = w · x + b
```

where:

- **w** = Weight vector
- **x** = Input feature vector
- **b** = Bias

The output function calculates the weighted sum of all input features and adds the bias.

---

# Threshold (Activation) Function

The computed value is passed through a threshold function to produce the final binary output.

### Heaviside Step Function

```
If f(x) ≥ 0
    y = 1

If f(x) < 0
    y = 0
```

### Sign Function

```
If f(x) ≥ 0
    y = +1

If f(x) < 0
    y = -1
```

The activation function converts the computed value into one of the two output classes.

---

# Summary

- Perceptron is a supervised learning algorithm.
- It is used for binary classification.
- It learns by updating weights during training.
- It predicts the class of new data during testing.
- It works only for linearly separable datasets.
- It forms the basic building block of Artificial Neural Networks (ANN).


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

print(df.head())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iris Data Set (3D)')
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Sepal width (cm)")
ax.set_zlabel("Petal length (cm)")

ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red', marker='o', s=40, edgecolor='red', label="Iris Setosa")
ax.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2], color='blue', marker='^', s=40, edgecolor='blue', label="Iris Versicolour")
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green', marker='x', s=40, label="Iris Virginica")

plt.legend(loc='upper left')
plt.show()

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self._misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self._misclassified_samples = []
        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self._misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)

import matplotlib.pyplot as plt

x_2d = df.iloc[0:100, 0:2].values
y_2d = df.iloc[0:100, 4].values

plt.figure(figsize=(8, 6))
plt.scatter(x_2d[:50, 0], x_2d[:50, 1], color='red', marker='o', label='Iris Setosa')
plt.scatter(x_2d[50:100, 0], x_2d[50:100, 1], color='blue', marker='x', label='Iris Versicolour')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('2D Scatter Plot of Sepal Length vs Sepal Width (Setosa vs Versicolour)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_perceptron = df.iloc[0:100, 0:2].values
y_perceptron_labels = df.iloc[0:100, 4].values

y_perceptron_transformed = np.where(y_perceptron_labels == 'Iris-setosa', 1, -1)

print("Original y labels (first 50):", y_perceptron_labels[:50])
print("Transformed y for Perceptron (first 50):", y_perceptron_transformed[:50])

x_perceptron[:, 0] = (x_perceptron[:, 0] - x_perceptron[:, 0].mean()) / x_perceptron[:, 0].std()
x_perceptron[:, 1] = (x_perceptron[:, 1] - x_perceptron[:, 1].mean()) / x_perceptron[:, 1].std()

print('\nStandardized X (first 5 rows):')
print(x_perceptron[:5])

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    x_perceptron,
    y_perceptron_transformed,
    test_size=0.25,
    random_state=0
)

print(f'\nShape of X_train: {X_train_p.shape}')
print(f'Shape of y_train: {y_train_p.shape}')
print(f'Shape of X_test: {X_test_p.shape}')
print(f'Shape of y_test: {y_test_p.shape}')

perceptron_classifier = Perceptron(learning_rate=0.01)
perceptron_classifier.fit(X_train_p, y_train_p)

y_pred_p = perceptron_classifier.predict(X_test_p)
accuracy = accuracy_score(y_test_p, y_pred_p)

print(f"Perceptron accuracy on the test set: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(perceptron_classifier._misclassified_samples) + 1),
         perceptron_classifier._misclassified_samples,
         marker='o', linestyle='-')
plt.title('Misclassified Samples per Epoch During Perceptron Training (Iris)')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassified Samples')
plt.grid(True)
plt.show()

```

# OUTPUT:

## 5 - Samples:
<img width="333" height="102" alt="{518B8C53-CC43-4A1C-9F46-75EB817FA365}" src="https://github.com/user-attachments/assets/3edf05a3-c139-412c-b25c-e644946c1762" />


## 3D - Visualization:
<img width="638" height="658" alt="image" src="https://github.com/user-attachments/assets/f22d6d5c-b5bf-40d4-87e4-8e31d9f7c660" />

## 2D - Visualization:
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/bd9fbc1b-3672-41ae-86e5-a02eccf16690" />

## Error vs Epoch:
<img width="678" height="470" alt="image" src="https://github.com/user-attachments/assets/d168206d-2c4e-42bd-a42b-e9fa140489bb" />

## Accuracy:
<img width="475" height="26" alt="{D91C25B2-B51A-4694-91F4-3239AC7B83FD}" src="https://github.com/user-attachments/assets/aad3f9f4-3a73-49d6-bdcd-8cdee449a285" />

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
