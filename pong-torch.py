import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
import cv2

H = 200
D = 80 * 65
C = 3
gamma = 0.99
batch_size = 50
learning_rate = 1e-3

render = False
resume = False
model_name = 'pong-torch.model'

class PolicyNet(nn.Module):

    def __init__(self):
        super(PolicyNet, self).__init__()

        self.hidden_layer = nn.Linear(D, H)
        self.output_layer = nn.Linear(H, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


def process(I):
    I = I[35:195, 15:-15]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    #cv2.imshow('pong-vision',I.astype(np.float))
    #cv2.waitKey(1)
    return I.astype(np.float).ravel()


def discount_rewards(reward_log):
    discount = 0
    discounted_rewards = np.zeros_like(reward_log)
    for idx in reversed(range(0, discounted_rewards.size)):
        if reward_log[idx] != 0: discount = 0
        discount = gamma * discount + reward_log[idx]
        discounted_rewards[idx] = discount
    return discounted_rewards


def main():
    env = gym.make('Pong-v0')
    observation = env.reset()
    policy = PolicyNet()
    if resume: policy.load_state_dict(torch.load(model_name))
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

    state_pool = []
    action_pool = []
    reward_pool = []
    prob_pool = []
    steps = 0
    prev_obs = None
    running_reward = None
    reward_sum = 0
    game = 0

    while True:
        if render: env.render()

        curr_obs = process(observation)
        x_diff = curr_obs - prev_obs if prev_obs is not None else np.zeros(D)
        prev_obs = curr_obs
        x = Variable(torch.from_numpy(x_diff).float())

        probs = policy(x)
        distribution = Bernoulli(probs)
        action = distribution.sample()  # returns 0 or 1
        log_prob = distribution.log_prob(action)  # cross entropy for a sample in a given distribution

        state_pool.append(x)
        action_pool.append(action)
        prob_pool.append(log_prob)

        action = 2 if action.data.numpy().astype(int)[0] == 1 else 3

        observation, reward, done, _ = env.step(action)

        reward_pool.append(reward)
        reward_sum += reward

        if done:
            steps += 1
            discounted_rewards = np.array(discount_rewards(reward_pool)).ravel()

            # https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
            # http://karpathy.github.io/2016/05/31/rl/ search "More general advantage functions"
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            if steps > 0 and steps % batch_size == 0:
                optimizer.zero_grad()
                policy_loss = []

                for log_prob, dis_reward in zip(prob_pool, discounted_rewards):
                    policy_loss.append(-log_prob * dis_reward)

                policy_loss = torch.cat(policy_loss).sum()

                policy_loss.backward()
                optimizer.step()

                state_pool = []
                action_pool = []
                reward_pool = []
                prob_pool = []

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f'resseting env. episode reward total was ${reward_sum} running mean: #{running_reward}')
            torch.save(policy.state_dict(), model_name)
            reward_sum = 0
            prev_obs = None
            observation = env.reset()
            game = 0

        if reward != 0:
            game += 1
            print(f"episode: {steps}, game: ${game} reward {reward} {'😋' if reward == 1 else ''}")

if __name__ == '__main__':
    main()