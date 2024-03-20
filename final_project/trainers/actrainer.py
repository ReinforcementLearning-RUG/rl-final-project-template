import torch
from torch import nn
from torchrl.data import ReplayBuffer

from final_project.models.twoheadedmlp import TwoHeadedMLP
from final_project.trainers.abstracttrainer import Trainer
from final_project.util.device import fetch_device

""""
IMPORTANT: in pseudocode gradient ascent is performed. But PyTorch automatic differentiation
facilities perform gradient descent by default. Therefore, you should reverse the signs to turn gradient ascent
in the pseudocode to gradient descent.
"""


class ACTrainer(Trainer):
    """
    One-step Actor-Critic Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(self,
                 buf: ReplayBuffer,
                 actor_model: TwoHeadedMLP,
                 critic_model: nn.Module,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float):
        """
        Initialize the Actor-Critic Trainer.

        :param buf: ReplayBuffer for storing experiences.
        :param actor_model: The actor model (policy).
        :param critic_model: The critic model (value function).
        :param learning_rate_actor: Learning rate for the actor.
        :param learning_rate_critic: Learning rate for the critic.
        :param discount_factor: Discount factor for future rewards.
        """
        self.device = fetch_device()
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.discount_factor = discount_factor
        self.I = 1  # Same I as in pseudocode, please give this a better name.

        self.buf = buf

        # Optimizes policy parameters
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=learning_rate_actor)
        # Optimizes critic parameters
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=learning_rate_critic)

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=1)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def train(self) -> None:
        pass