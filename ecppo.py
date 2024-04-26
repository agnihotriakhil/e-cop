
"""Implementation of the ECPPO algorithm."""

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.algorithms.on_policy.penalty_function.p3o import P3O
from omnisafe.utils import distributed


# @registry.register
class ECPPO(P3O):
    """
    Implementation of the ECPPO algorithm, to be run with omnisafe CLI
    """

    def _init_log(self) -> None:
        
        super()._init_log()
        self._logger.register_key('Loss/Loss_pi_cost', delta=True)


    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        time_step: int,
    ) -> torch.Tensor:
        r"""Computing pi reward loss.

        Args:
            obs (torch.Tensor): The ``observations`` sampled from buffer of the previous episode.
            act (torch.Tensor): The ``actions`` sampled from buffer of the previous episode.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer of the previous episode.
            adv_r (torch.Tensor): The ``reward_advantage`` here.
            time_step (int): The time step in the current episode to update the actor at.

        Returns:
            The loss of pi/actor.
        """
        
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        ratio_clipped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.epsilon,
            1 + self._cfgs.algo_cfgs.epsilon,
        )

        loss = self._episodic_reward_loss(distribution, adv_r, ratio, ratio_clipped, time_step)
        # loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        return loss


    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        time_step : int,
    ) -> torch.Tensor:
        r"""Compute the performance of cost after each episode. Input ``obs'' is sampled from the trajectory buffer.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer of the previous episode. 
            act (torch.Tensor): The ``action`` sampled from buffer of the previous episode.
            logp (torch.Tensor): The ``log probability`` of actions sampled from buffer of the previous episode.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer of the previous episode.
            time_step (int): The time step at which to update the new policy.
        Returns:
            The loss of the reward performance.
        """

        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        ratio_clipped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.epsilon,
            1 + self._cfgs.algo_cfgs.epsilon,
        )
        surr_cadv = self._get_surr_cadv(distribution, adv_c, ratio, ratio_clipped, time_step)
        Jc = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        lamb_ = self._update_lamb(Jc)
        coeff_ratio = lamb_ / self._cfgs.algo_cfgs.beta
        loss_cost = lamb_ * F.relu(surr_cadv + Jc) + self._cfgs.algo_cfgs.beta/2 * ((F.relu(surr_cadv + Jc + coeff_ratio))**2 - coeff_ratio**2)
        self._logger.store({'Loss/Loss_pi_cost': loss_cost.mean().item()})
        return loss_cost.mean()


    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        time_step: int,
    ) -> None:
        """Update policy network under a double for loop. This loop runs for each policy in the episode.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            logp (torch.Tensor): ``log_p`` stored in buffer.
            adv_r (torch.Tensor): ``reward_advantage`` stored in buffer.
            adv_c (torch.Tensor): ``cost_advantage`` stored in buffer.
            time_step (int):
        """

        loss_reward = self._loss_pi(obs, act, logp, adv_r, time_step)
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c, time_step)

        loss = loss_reward + loss_cost

        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
