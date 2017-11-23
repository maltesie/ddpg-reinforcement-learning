from rllab.algos.reps import REPS
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = normalize(GymEnv("Pendulum-v0"))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = ZeroBaseline(env_spec=env.spec)

    algo = REPS(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=8000, #tried 1,50,100,1000,2000,4000,8000
        max_path_length=env.horizon,
        n_itr=50,
        epsilon=0.5, #tried 0.2,0.4,0.5,0.6,0.8
        L2_reg_dual=1e-5,  #tried 1e-5,1e-8,1e-2
        L2_reg_loss=0.,
        discount=0.99,
        step_size=0.1, #tried 0.01,0.1
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
