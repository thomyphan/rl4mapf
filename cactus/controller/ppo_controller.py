from cactus.controller.a2c_controller import A2CController
from cactus.utils import get_param_or_default
from cactus.constants import *
from torch.distributions import Categorical

class PPOController(A2CController):
    
    def __init__(self, params) -> None:
        params[CLIP_RATIO] = get_param_or_default(params, CLIP_RATIO, 0.1)
        params[UPDATE_ITERATIONS] = get_param_or_default(params, UPDATE_ITERATIONS, 4)
        super(PPOController, self).__init__(params)

    def policy_loss(self, advantage, probs, action, old_probs=None):
        m1 = Categorical(probs)
        logprobs = m1.log_prob(action)
        m2 = Categorical(old_probs)
        old_logprobs = m2.log_prob(action)
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # Calculate Surrogate Losses 
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
        # final loss of clipped objective PPO
        return -torch.min(surr1, surr2)