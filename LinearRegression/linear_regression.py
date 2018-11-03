import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def accuracy_score(pred_y, test_y):
    e = np.sum((pred_y - test_y) ** 2)
    t = np.sum((test_y - np.sum(test_y) / int(n / 4)) ** 2)
    return 1 - e / t


np.random.seed(100)

# function params
a = 2.4
b = 3.5
noise = 5
n = 100

# regression params
a0 = 0
b0 = 0
alpha = 0.001

# data sets
train_x = np.random.randint(10,50,(n,1))
test_x = np.random.randint(10,50,(int(n / 4),1))
train_y = np.full((n,1), a * train_x + b + np.random.rand(n,1) * noise)
test_y = np.full((int(n / 4),1), a * test_x + b + np.random.rand(int(n / 4),1) * noise)

# linear regression
for i in range(10000):
    y = a0 * train_x + b0
    e = y - train_y
    mse = np.sum(e * e) / n
    b0 = b0 - 2 * alpha * np.sum(e) / n
    a0 = a0 - 2 * alpha * np.sum(e * train_x) / n

# prediction
pred_y = b0 + a0 * test_x


# r2_scores
print("a:" + str(a0) + "\nb:" + str(b0) + "\nr2_score:"
      + str(r2_score(pred_y, test_y)) + "\nmy_r2_score:" + str(accuracy_score(pred_y, test_y)))

# plotting
plt.plot(test_x, test_y, 'ro', label="test")
plt.plot(train_x, train_y, 'go', label="train")
plt.plot(test_x, b0 + a0 * test_x, 'b', label="prediction")
plt.axis([0, 60, 0, 150])
plt.legend()
plt.show()
