from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


##
# HOPE ping-pong runners (shared hyper-parameters from cfg/algo/ppo.yaml, as the A3's agents/ppo.py).
# experiment_name must match cfg/task/HOPEPingPongUnitreeG1.yaml so both launch paths log together.
##

from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs  # noqa: E402

_HOPE_KW = runner_kwargs(load_ppo_params(), "unitree_g1_hope")


@configclass
class HOPEUnitreeG1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _HOPE_KW["num_steps_per_env"]
    max_iterations = _HOPE_KW["max_iterations"]
    save_interval = _HOPE_KW["save_interval"]
    experiment_name = _HOPE_KW["experiment_name"]
    empirical_normalization = _HOPE_KW["empirical_normalization"]
    policy = _HOPE_KW["policy"]
    algorithm = _HOPE_KW["algorithm"]


@configclass
class HitterPingPongUnitreeG1PPORunnerCfg(HOPEUnitreeG1PPORunnerCfg):
    experiment_name = "unitree_g1_hitter_pingpong"
