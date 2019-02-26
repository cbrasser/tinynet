import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork

df = pd.read_csv('data/train.csv')

# Features
df_X = df.loc[:100,'x1':'x10']
# Labels
df_Y = df.loc[:100,'y']

# Convert to numpy arrays, because fast
X = np.asarray(df_X)
y = np.asarray([df_Y])
print(y.shape)
nn = NeuralNetwork(10,1)

for i in range(0,1000):
    nn.train(X,y)
    print (f'Loss: {np.mean(np.square(y - nn.forward(X)))}')


print(nn.forward(np.asarray([1,2,3,4,5,6,7,8,9,10])))
