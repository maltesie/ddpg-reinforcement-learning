import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pa

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
        ax2.legend(loc='upper right')
    plt.show()

def rolling_mean(data, n):
    results = np.empty((data.shape[0]-n, data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[0]-n):
            results[j,i] = np.mean(data[j:j+n,i])
    return results
            
#plot_averaged_np_dumps('CartPole-v1', 'average-nn-model-10-eval', False, True, 10, 10)
opts = {'average-nn-model-10-eval':{'length':1100, 'model':True, 'freq':10, 'marker':'r-', 'label':'RW every 10 episodes'}, 
        'average-nn-model-5-eval':{'length':1200, 'model':True, 'freq':5, 'marker':'b-', 'label':'RW every 5 episodes'}, 
        'average-no-model-no-eval':{'length':1000, 'model':False, 'marker':'g-', 'label':'no model'},
        'average-gp-model-10-eval':{'length':1100, 'model':True, 'freq':10, 'marker':'r-', 'label':'RW every 10 episodes'}, 
        'average-gp-model-5-eval':{'length':1200, 'model':True, 'freq':5, 'marker':'b-', 'label':'RW every 5 episodes'}}

def mse_plots(env, alg):
    for prefix in ['average-'+alg+'-model-10-eval', 'average-'+alg+'-model-5-eval']:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8.5, 4.5)
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        plot_data = np.zeros((opts[prefix]['length'],2))
        for filename in [prefix+'_'+str(i)+'.npy' for i in range(5)]:
            with open('./results/'+env+'/np_dumps/'+filename, 'rb') as f:
                plot_data += np.load(f)[:opts[prefix]['length']]
        plot_data /= 5
        plot_data = rolling_mean(plot_data[np.asarray([not i%opts[prefix]['freq'] == 0 for i in range(opts[prefix]['length'])])],20)
        ax1.plot(plot_data[:,0], 'r-', markersize=5, label=u'reward in model')
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(plot_data[:,1], 'g-', markersize=5, label=u'model error')
        ax2.set_ylabel('mean eucleadian distance', color='g')
        ax2.legend(loc='upper right')
        fig.savefig(env+'_'+prefix+'_modelerror.png')

def comparison_plots(env, alg):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward')
    fig.set_size_inches(8.5, 4.5)
    max_eps = []
    for prefix in ['average-'+alg+'-model-10-eval', 'average-'+alg+'-model-5-eval', 'average-no-model-no-eval']:
        plot_data = np.zeros((opts[prefix]['length'],2))
        for filename in [prefix+'_'+str(i)+'.npy' for i in range(5)]:
            with open('./results/'+env+'/np_dumps/'+filename, 'rb') as f:
                plot_data += np.load(f)[:opts[prefix]['length']]
        plot_data /= 5
        if opts[prefix]['model']:
            plot_data = rolling_mean(plot_data[::opts[prefix]['freq']+1],10)
            max_eps.append(np.argmax(plot_data[:,0]))
        else: 
            plot_data = rolling_mean(plot_data,10)
            max_eps.append(np.argmax(plot_data[:,0]))
        ax1.plot(plot_data[:,0], opts[prefix]['marker'], markersize=5, label=opts[prefix]['label'])
    fig.suptitle('Maximum reached after:')
    ax1.set_title('red: {}, blue: {}, green: {}'.format(*max_eps))
    #ax1.set_xticklabels([10,210,410,610,810,1010])
    ax1.legend(loc='lower right')
    fig.savefig(env+'_'+alg+'_comp.png')

def model_accuracy_plots():
    pass

def plotting_explanation():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('episode')
    fig.set_size_inches(8.5, 4.5)
    ax1.plot(np.zeros(100), 'r-', markersize=5, label=u'model based every 10 episodes')
    ax1.plot(np.zeros(100), 'b-', markersize=5, label=u'model based every 5 episodes')
    ax1.plot(np.zeros(100), 'g-', markersize=10, label=u'model free')
    for i in range(20):
        ax1.add_patch(pa.Arrow(i*5.0, 0.01, 0., -0.01, width=2,facecolor="blue"))
        #ax1.arrow( 0.01, i*5.0, 0.0, i*5.0, shape='full', head_width=0.01, color='b' )
    for i in range(10):
        ax1.add_patch(pa.Arrow(i*10., -0.01, 0., 0.01, width=2,facecolor="red"))
        #ax1.arrow( -0.01, i*10., 0.0, i*10.0, shape='full', head_width=0.01, color='r' )    
    ax1.set_yticks([])
    ax1.legend(loc='lower right')
    fig.savefig('plotting_explanation.png')


plotting_explanation()

comparison_plots('Pendulum-v0', 'nn')
comparison_plots('Pendulum-v0', 'gp')
comparison_plots('CartPole-v1', 'gp')
comparison_plots('CartPole-v1', 'nn')

mse_plots('CartPole-v1', 'nn')
mse_plots('CartPole-v1', 'gp')
mse_plots('Pendulum-v0', 'gp')
mse_plots('Pendulum-v0', 'nn')