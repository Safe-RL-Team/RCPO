from collections import OrderedDict
import torch

half_cheetah_params = OrderedDict(
    [
        ("batch_size", 64),
        ("clip_range", 0.1),
        ("ent_coef", 0.000401762),
        ("gae_lambda", 0.92),
        ("gamma", 0.98),
        ("learning_rate", 2.0633e-05),
        ("max_grad_norm", 0.8),
        ("n_epochs", 20),
        ("n_steps", 512),
        ("policy", "MlpPolicy"),
        (
            "policy_kwargs",
            dict(
                log_std_init=-2,
                ortho_init=False,
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            ),
        ),
        ("vf_coef", 0.58096),
    ]
)
