import numpy as np
import matplotlib.pyplot as plt

def plot_csv_tensorboard():
    evaluation = np.loadtxt('results/run_.,tag_Evaluation.csv', delimiter=',', skiprows=1)
    reward = np.loadtxt('results/run_.,tag_Reward.csv', delimiter=',', skiprows=1)
    plt.plot(reward[:,1], reward[:,2], 'r-', markersize=5, label=u'within model')
    plt.plot(evaluation[:,1], evaluation[:,2],  'b-', markersize=5, label=u'in environment')
    print(reward)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    plt.show()
    
def plot_np_dumps(env_name, prefix, plot_mse, use_model, n, ep_real):
    for filename in [prefix+'_'+str(i)+'.npy' for i in range(n)]:
        with open('./results/'+env_name+'/np_dumps/'+filename, 'rb') as f:
            plot_data = np.load(f)
        if use_model: plot_data = rolling_mean(plot_data[::ep_real+1],10)
        else: plot_data = rolling_mean(plot_data[:210],10)
        fig, ax1 = plt.subplots()
        ax1.plot(plot_data[:,0], 'r-', markersize=5, label=u'within model')
        ax1.plot(plot_data[:,1],  'b-', markersize=5, label=u'in environment')
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        ax1.legend(loc='upper left')
        if plot_mse:
            ax2 = ax1.twinx()
            ax2.plot(plot_data[:,2], 'g-', markersize=5, label=u'model mse')
            ax2.set_ylabel('max. mse', color='g')
            ax2.legend(loc='up')
        plt.show()

def plot_averaged_np_dumps(env_name, prefix, plot_mse, use_model, n, ep_real):
    if not use_model: plot_data = np.zeros((1000,2))
    #else: plot_data = np.zeros((int(1000+100*10/ep_real),2))
    else: plot_data = np.zeros((int(1200),2))
    for filename in [prefix+'_'+str(i)+'.npy' for i in range(n)]:
        with open('./results/'+env_name+'/np_dumps/'+filename, 'rb') as f:
            plot_data += np.load(f)
    plot_data /= n
    if use_model:
        plot_data = rolling_mean(plot_data[::ep_real+1],10)
    else: plot_data = rolling_mean(plot_data,10)
    fig, ax1 = plt.subplots()
    ax1.plot(plot_data[:,0], 'r-', markersize=5, label=u'within model')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward')
    ax1.legend(loc='upper left')
    if plot_mse:
        ax2 = ax1.twinx()
        ax2.plot(plot_data[:,1], 'g-', markersize=5, label=u'model mse')
        ax2.set_ylabel('mean rmse', color='g')
        ax2.legend(loc='up')
    plt.show()

def rolling_mean(data, n):
    results = np.empty((data.shape[0]-n, data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[0]-n):
            results[j,i] = np.mean(data[j:j+n,i])
    return results
            
plot_averaged_np_dumps('CartPole-v1', 'average-nn-model-10-eval', False, True, 5, 10)