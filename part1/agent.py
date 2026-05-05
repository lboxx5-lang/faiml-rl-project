import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    """
    Compute discounted Monte Carlo returns:

        G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...

    Args:
        r: tensor of rewards, shape (T,)
        gamma: discount factor

    Returns:
        discounted_r: tensor, shape (T,)
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0.0

    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
        Actor network.

        Input:
            state s

        Output:
            mean of Gaussian policy pi(a|s)
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration.
        # softplus keeps sigma positive.
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
        Critic network.

        Input:
            state s

        Output:
            scalar value V(s)
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Returns:
            normal_dist: Gaussian action distribution
            state_value: V(s)
        """

        """
        Actor forward
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma) + 1e-6
        normal_dist = Normal(action_mean, sigma)

        """
        Critic forward
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device="cpu", algorithm="reinforce"):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.gamma = 0.99

        """
        Available algorithms:
            - reinforce
            - reinforce_baseline
            - actor_critic
        """
        self.algorithm = algorithm

        # Constant baseline for REINFORCE with baseline.
        self.baseline = 0.0
        self.baseline_alpha = 0.9

        # Weight of critic loss in Actor-Critic.
        self.critic_coef = 0.5

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = (
            torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        )

        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)

        next_states = (
            torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        )

        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        done = torch.Tensor(self.done).to(self.train_device)

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        returns = discount_rewards(rewards, self.gamma)

        if self.algorithm == "reinforce":
            """
            REINFORCE without baseline:

                loss = - sum_t log pi(a_t|s_t) G_t
            """

            signal = returns

            # Numerical stabilization.
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

            policy_loss = -(action_log_probs * signal.detach()).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == "reinforce_baseline":
            """
            REINFORCE with constant baseline:

                loss = - sum_t log pi(a_t|s_t) (G_t - b)
            """

            episode_return = returns[0].item()

            self.baseline = (
                self.baseline_alpha * self.baseline
                + (1.0 - self.baseline_alpha) * episode_return
            )

            signal = returns - self.baseline

            # Numerical stabilization.
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

            policy_loss = -(action_log_probs * signal.detach()).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == "actor_critic":
            """
            Actor-Critic.

            Critic estimates:
                V(s_t)

            Advantage:
                A_t = G_t - V(s_t)

            Actor loss:
                L_actor = - sum_t log pi(a_t|s_t) A_t

            Critic loss:
                L_critic = MSE(V(s_t), G_t)

            Total loss:
                L = L_actor + critic_coef * L_critic
            """

            _, values = self.policy(states)

            advantages = returns - values.detach()

            # Numerical stabilization of advantage.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages).sum()
            critic_loss = F.mse_loss(values, returns.detach())

            loss = actor_loss + self.critic_coef * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_action(self, state, evaluation=False):
        """
        Convert state into action.

        During training:
            sample action from Gaussian policy.

        During evaluation:
            use mean action.
        """

        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:
            action = normal_dist.mean
            return action, None

        action = normal_dist.sample()

        # Hopper has 3 continuous action components.
        # log pi(a|s) = sum_i log pi(a_i|s)
        action_log_prob = normal_dist.log_prob(action).sum()

        return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
