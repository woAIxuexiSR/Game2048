'''
maxvalue ~ 386w
'''

from game_2048 import Game2048

import numpy as np
from collections import deque

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

HEIGHT, WIDTH = 4, 4
DX = ((-1, 0), (1, 0), (0, -1), (0, 1))

def preprocess(state):
    state[np.where(state == 0)] = 1
    return np.log2(state)

def forbid(state):
    mask = np.array([1, 1, 1, 1])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if state[i][j] != 0:
                for k in range(4):
                    ii, jj = i + DX[k][0], j + DX[k][1]
                    if ii >= 0 and ii < HEIGHT and jj >= 0 and jj < WIDTH:
                        if state[i][j] == state[ii][jj] or state[ii][jj] == 0:
                            mask[k] = 0
    return 1 - mask

class DQN(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, (1, 1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 4, 4)
        x = self.conv(x)
        x = x.reshape(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x


class Agent:
    def __init__(self, obs_space, act_space, device, discount_rate=0.95, eps_min=0.05, eps_max=0.95, eps_decay_steps=1000000):
        self.net = DQN(act_space).to(device)
        self.device = device
        self.act_space = act_space
        self.discount_rate = discount_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        self.step = 1
        self.rng = np.random.default_rng()

        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)

    def best_act(self, state):
        self.net.eval()
        with torch.no_grad():
            qvalues = self.net(torch.from_numpy(state).to(self.device, dtype=torch.float32)).squeeze()
        mask = torch.from_numpy(forbid(state)).to(self.device)
        qvalues[mask == 0] = -np.inf
        return torch.argmax(qvalues).item()

    def act(self, state):
        mask = forbid(state)
        if np.sum(mask) == 0: return 0

        eps = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * self.step / self.eps_decay_steps)
        if self.rng.random() < eps:
            return np.random.choice(np.where(mask != 0)[0])
        else :
            return self.best_act(state)

    def train(self, data):
        s0 = torch.tensor(data[0], dtype=torch.float32).to(self.device)
        a = torch.tensor(data[1], dtype=torch.int64).view(-1, 1)
        r = torch.tensor(data[2], dtype=torch.float32).view(-1, 1)
        s1 = torch.tensor(data[3], dtype=torch.float32).to(self.device)
        d = torch.tensor(data[4], dtype=torch.int32).view(-1, 1)

        qvalues = self.net(s0).cpu()
        next_qvalues = self.net(s1).cpu()

        y_pred = qvalues.gather(1, a)
        y_true = r + self.discount_rate * d * torch.max(next_qvalues, dim=1)[0].view(-1, 1)

        loss = F.mse_loss(y_pred, y_true)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step += 1

        return loss

    def save_model(self, path):
        # self.net.to(torch.device("cpu"))
        torch.save(self.net, path)
        # torch.save(self.net.state_dict(), path)

if __name__ == "__main__":

    train_start = 20
    train_interval = 10
    train_times = 10
    batch_size = 32

    print_interval = 500

    episode_num = 500000
    test_episode_num = 1000

    env = Game2048()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        16, 
        4,
        device
        )
    writer = SummaryWriter()

    avg_reward = 0
    buffer = deque([], maxlen=1000)
    for episode in range(episode_num):

        state = env.reset()
        state = preprocess(state)
        sum_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = preprocess(next_state)

            buffer.append((state, action, reward, next_state, 1 - done))

            state = next_state
            sum_reward += reward

            if done :
                break

        avg_reward += sum_reward

        if episode % print_interval == print_interval - 1:
            print("episode {}, average reward : {}".format(episode, avg_reward / print_interval))
            writer.add_scalar("average reward", avg_reward, episode)
            avg_reward = 0
        
        if episode < train_start or episode % train_interval != 0:
            continue
        
        for _ in range(train_times):
            indices = np.random.permutation(len(buffer))[:batch_size]
            data = [[], [], [], [], []]
            for idx in indices:
                memory = buffer[idx]
                for col, value in zip(data, memory):
                    col.append(value)
            
            loss = agent.train(data)
            # print("loss : {}".format(loss))
    writer.close()


    avg_reward = 0
    for episode in range(test_episode_num):
        
        state = env.reset()
        state = preprocess(state)
        sum_reward = 0

        while True:
            action = agent.best_act(state)
            next_state, reward, done = env.step(action)
            next_state = preprocess(next_state)
            state = next_state
            sum_reward += reward
            
            if done:
                break

        avg_reward += sum_reward

    agent.save_model("test.pth")

    avg_reward /= test_episode_num
    print("test average reward : {}".format(avg_reward))