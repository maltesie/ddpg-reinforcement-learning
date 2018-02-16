import numpy as np

test = np.arange(100)[::6]
print(test)

for i in range(100):
	if i%5 == 0 : print(i)