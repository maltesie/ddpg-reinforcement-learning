import numpy as np
import matplotlib.pyplot as plt

progress = np.loadtxt('progress.csv', skiprows=1, delimiter=',')
l1 , = plt.plot(progress[:,0], progress[:,1], label='AvgDiscountedRet')
l2 , = plt.plot(progress[:,0], progress[:,3], label='AvgDiscountedRetStd')
l3 , = plt.plot(progress[:,0], progress[:,12], label='AverageReturn')
plt.legend([l2, l1, l3], ['Std - Average Discounted Return', 'Average Discounted Return', 'Average Return'])
