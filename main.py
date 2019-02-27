import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork

df = pd.read_csv('data/train.csv')

# Features
df_X = df.loc[:100,'x1':'x10']
# Labels
df_Y = df.loc[:100,'y']

test_y = []
test_x = []
for i in range(100):
    a = np.random.randint(0,10)
    b = np.random.randint(0,10)
    c = np.random.randint(0,10)
    test_x.append([a,b,c])
    value = 3 * a + 5*b - 4*c
    if value > 15:
        test_y.append(1)
    else:
        test_y.append(0)


print(test_y)
test_X = np.asarray(test_x)
test_y = np.asarray([test_y])

# Convert to numpy arrays, because fast
X = np.asarray(df_X)
y = np.asarray([df_Y])
print(test_X.shape)
print(y.shape)
nn = NeuralNetwork(3,1)

for i in range(0,1000):
    nn.train(test_X,test_y)
    print (f'Loss: {np.mean(np.square(test_y - nn.forward(test_X)))}')


print(nn.forward(np.asarray([2,2,4])))
print(nn.forward(np.asarray([20,2,4])))
