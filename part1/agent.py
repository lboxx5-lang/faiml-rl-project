"""Policy and learning agent for the Hopper environment.

Supports two algorithms via the `algorithm` flag:
  - 'reinforce': Monte Carlo policy gradient, with optional constant baseline
  - 'ac':        one-step bootstrapped Actor-Critic (TD(0))
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    """Compute G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... for each t."""
    discounted = torch.zeros_like(r)
    running = 0.0
    for t in reversed(range(r.size(-1))):
        running = running * gamma + r[t]
        discounted[t] = running
    return discounted


class Policy(torch.nn.Module):
    """Actor-Critic network.

    Actor: state -> Gaussian over actions (mean from MLP, sigma free param).
    Critic: state -> scalar V(s). Used only by the AC algorithm.
    """

    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor MLP: 11 -> 64 -> 64 -> 3 (action mean)
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Gaussian std: one trainable scalar per action dim, kept positive by softplus.
        self.sigma_activation = F.softplus
        self.sigma = torch.nn.Parameter(torch.zeros(action_space) + 0.5)

        # Critic MLP: 11 -> 64 -> 64 -> 1 (state value)
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Actor branch -> Normal(mean, sigma)
        a = self.tanh(self.fc1_actor(x))
        a = self.tanh(self.fc2_actor(a))
        mean = self.fc3_actor_mean(a)
        sigma = self.sigma_activation(self.sigma)
        dist = Normal(mean, sigma)

        # Critic branch -> V(s)
        c = self.tanh(self.fc1_critic(x))
        c = self.tanh(self.fc2_critic(c))
        value = self.fc3_critic(c)

        return dist, value


class Agent(object):
    """On-policy learner. Collects one full episode, then runs one gradient step."""

    def __init__(self, policy, device="cpu", baseline=0.0, algorithm="reinforce"):
        assert algorithm in ("reinforce", "ac")

        self.device = device
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.baseline = baseline  # constant baseline for REINFORCE (0.0 = none)
        self.algorithm = algorithm

        # Per-episode rollout buffers, cleared after each update.
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def get_action(self, state, evaluation=False):
        """Sample an action (training) or take the Gaussian mean (evaluation)."""
        x = torch.from_numpy(state).float().to(self.device)
        dist, _ = self.policy(x)  # critic value ignored when picking actions
        if evaluation:
            return dist.mean, None
        action = dist.sample()
        # Independent action dims: sum log-probs over the action vector.
        log_prob = dist.log_prob(action).sum()
        return action, log_prob

    def store_outcome(self, state, next_state, log_prob, reward, done):
        """Append one transition to the rollout buffers."""
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def update_policy(self):
        """Build the loss for the just-finished episode and do one Adam step."""
        # Stack buffers into tensors of shape (T,) or (T, state_dim).
        log_probs = torch.stack(self.action_log_probs).to(self.device).squeeze(-1)
        states = torch.stack(self.states).to(self.device).squeeze(-1)
        next_st = torch.stack(self.next_states).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards).to(self.device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.device)

        # Empty buffers immediately, so a crash later cannot leave stale data.
        self.states, self.next_states = [], []
        self.action_log_probs, self.rewards, self.done = [], [], []

        if self.algorithm == "reinforce":
            # ---- REINFORCE ----
            # Monte Carlo returns minus optional constant baseline.
            returns = discount_rewards(rewards, self.gamma)
            advantages = returns - self.baseline
            # Policy gradient: maximize E[log pi(a|s) * A] -> minimize the negative.
            loss = -(log_probs * advantages.detach()).sum()

        else:
            # ---- Actor-Critic (one-step TD bootstrap) ----
            # Recompute V(s) and V(s') with current critic weights so we have
            # a fresh gradient path for the critic loss.
            _, values = self.policy(states)
            values = values.squeeze(-1)
            _, next_values = self.policy(next_st)
            next_values = next_values.squeeze(-1)

            # TD target: r + gamma * V(s'), zeroed out on terminal transitions
            # (since V(terminal) == 0).
            not_done = 1.0 - done
            td_target = rewards + self.gamma * next_values.detach() * not_done

            # Advantage = TD error. Detach target so critic chases V, not the other way.
            advantage = td_target - values

            actor_loss = -(log_probs * advantage.detach()).sum()
            critic_loss = F.mse_loss(values, td_target.detach(), reduction="sum")
            loss = actor_loss + critic_loss

        # Single Adam step on both actor and critic parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
