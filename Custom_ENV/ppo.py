import numpy as np
import torch
import gymnasium as gym
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam
from torch.nn import MSELoss
import wandb

from neural_network import FeedForwardNN


class PPO:
    def __init__(self, env: gym.Env, obs_enc_dim: int) -> None:
        # Environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.obs_enc_dim = obs_enc_dim
        self.act_dim = env.action_space.n
        # self.act_dim = env.action_space.shape[0]

        # Hyperparameters
        self._init_hyperparameters()

        # Initialize a new wandb run
        wandb.init(
            project="my-ppo-project",
            config={
                "timesteps_per_batch": self.timesteps_per_batch,
                "max_timesteps_per_episode": self.max_timesteps_per_episode,
                "gamma": self.gamma,
                "n_updates_per_iteration": self.n_updates_per_iteration,
                "clip": self.clip,
                "lr": self.lr,
            },
        )

        # ALG STEP 1
        # Actor and critic networks
        """
        self.actor = FeedForwardNN(self.obs_enc_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_enc_dim, self.act_dim)
        """
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        # Observations encoder
        # self.obs_enc = FeedForwardNN(self.obs_dim, self.obs_enc_dim)

        # Network optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # self.obs_enc_optim = Adam(self.obs_enc.parameters(), lr = self.lr)

        # Multivariate Normal Stats
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        # Default hyperparameter values - NEED TO CHANGE
        self.timesteps_per_batch = 100000
        self.max_timesteps_per_episode = 10000
        self.gamma = 0.95  # reward decay
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # recommended by PPO paper
        self.lr = 0.005

    def learn(self, n_steps):
        n = 0  # number of steps taken
        while n < n_steps:  # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5 - Calculate Advantage
            A_k = batch_rtgs - V.detach()

            # As found empirically by Eric Yu, using raw advantage values can lead to training instability,
            # so this project follows his advice and uses normalized advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # ALG STEP 6 & 7
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratio from PPO algorithm
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calculate surrogate losses from PPO algorithm
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate losses.  Use Actor loss to pass through to encoder
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = MSELoss()(V, batch_rtgs)
                # encoder_loss = (-torch.min(surr1, surr2)).mean()

                # Perform backward propogation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Perform backward propogation for actor network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Calculate additional metrics
                episode_rewards = batch_rtgs.sum().item()
                episode_lengths = torch.tensor(batch_lens).float().mean().item()

                entropy = (
                    MultivariateNormal(self.actor(batch_obs), self.cov_mat)
                    .entropy()
                    .mean()
                    .item()
                )
                learning_rate = self.actor_optim.param_groups[0]["lr"]

                # Log metrics to wandb
                wandb.log(
                    {
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "episode_rewards": episode_rewards,
                        "episode_lengths": episode_lengths,
                        "entropy": entropy,
                        "learning_rate": learning_rate,
                        # Add any other metrics you want to track
                    }
                )

                # Calculate how many timesteps collected in batch
                n += np.sum(batch_lens)

                # Perform backward propogation for encoder network
                """
                self.obs_enc_optim.zero_grad()
                encoder_loss.backward()
                self.obs_enc_optim.step()
                """

    def rollout(self):
        # batch data
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards to go
        batch_lens = []  # episodic lengths in batch

        # Number of timesteps run this batch
        t = 0

        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []

            obs, _ = self.env.reset()
            obs = np.ravel(obs)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps run this batch
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                # print(action)
                obs, rew, done, _, _ = self.env.step(action)
                obs = np.ravel(obs)

                # Collect reward, action, and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # reshape as tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        # ALG STEP 4 - Compute rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # encode the observations and query the actor for mean action
        # obs = self.obs_enc(obs)
        mean = self.actor(obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample action from distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # return detached action and log prob
        return action.detach().numpy(), log_prob.detach().numpy()

    def compute_rtgs(self, batch_rews):
        # reawards-to-go per episode to return
        batch_rtgs = []

        # iterate through episode backwards
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # running reward
            for rew in reversed(ep_rews):
                discounted_reward = rew + (discounted_reward * self.gamma)
                batch_rtgs.insert(0, discounted_reward)

        # convert to tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        # print('rtgs', batch_rtgs.shape)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        # query critic network for value V for each obs in batch_obs after encoding
        # batch_obs = self.obs_enc(batch_obs)
        V = self.critic(batch_obs).squeeze()
        # print('eval', V.detach().shape)

        # get log probabilities
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
