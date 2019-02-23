import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR_ACTOR = 0.0001         # learning rate of the actor 
LR_CRITIC = 0.0003        # learning rate of the critic
WEIGHT_DECAY = 0   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        # Actor Network (w/ Target Network)
        self.actor_local1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer1 = optim.Adam(self.actor_local1.parameters(), lr=LR_ACTOR)
        self.actor_local2 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target2 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer2 = optim.Adam(self.actor_local2.parameters(), lr=LR_ACTOR)
        #critic_state_size = np.reshape(state_size, 48)
        # Critic Network (w/ Target Network)
        self.critic_local1 = Critic(state_size*2, action_size, random_seed).to(device)
        self.critic_target1 = Critic(state_size*2,action_size, random_seed).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_local2 = Critic(state_size*2, action_size, random_seed).to(device)
        self.critic_target2 = Critic(state_size*2, action_size, random_seed).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
       
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > 7000:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        #self.noise = OUNoise(self.action_size, random_seed,sigma=sigma)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local1.eval()
        with torch.no_grad():
            action1 = self.actor_local1(state[0]).cpu().data.numpy()
        self.actor_local1.train()
        if add_noise:
            action1 += self.noise.sample()
        self.actor_local2.eval()
        with torch.no_grad():
            action2 = self.actor_local2(state[1]).cpu().data.numpy()
        self.actor_local2.train()
        if add_noise:
            action2 += self.noise.sample()
        
        return np.vstack((np.clip(action1, -1, 1), np.clip(action2, -1, 1)))

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        statesforcritic = torch.reshape(states, (BATCH_SIZE, self.state_size*2))
        nextstatesforcritic = torch.reshape(next_states, (BATCH_SIZE, self.state_size*2))
        actionsforcritic = torch.reshape(actions, (BATCH_SIZE, self.action_size*2))
        nextstatesforactor = torch.split(nextstatesforcritic,self.state_size,1)
        statesforactor = torch.split(statesforcritic,self.state_size,1)
        actionsforactor = torch.split(actionsforcritic,self.action_size,1)
        rewardsforactor = torch.split(rewards,1,1)
        donesforactor = torch.split(dones,1,1)
      
        # --------------------------- update critic 1---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next1 = self.actor_target1(nextstatesforactor[0])
        actions_next2 = self.actor_target2(nextstatesforactor[1])
        actions_next = torch.cat((actions_next1, actions_next2), 1)
        
        Q_targets_next_1 = self.critic_target1(nextstatesforcritic, actions_next1)
        # Compute Q targets for current states (y_i)
        Q_targets1 = rewardsforactor[0] + (gamma * Q_targets_next_1 * (1 - donesforactor[0]))
        # Compute critic loss
        Q_expected1 = self.critic_local1(statesforcritic, actionsforactor[0])
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets1)
        # Minimize the loss
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()
        
        
        Q_targets_next_2 = self.critic_target2(nextstatesforcritic, actions_next2)
        # Compute Q targets for current states (y_i)
        Q_targets2 = rewardsforactor[1] + (gamma * Q_targets_next_2 * (1 - donesforactor[1]))
        # Compute critic loss
        Q_expected2 = self.critic_local2(statesforcritic, actionsforactor[1])
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets2)
            # Minimize the loss
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred1 = self.actor_local1(statesforactor[0])
        
        actor_loss1 = -self.critic_local1(statesforcritic, actions_pred1).mean()
        # Minimize the loss
        self.actor_optimizer1.zero_grad()
        actor_loss1.backward()
        self.actor_optimizer1.step()
        
        actions_pred2 = self.actor_local2(statesforactor[1])
        actor_loss2 = -self.critic_local2(statesforcritic, actions_pred2).mean()
        # Minimize the loss
        self.actor_optimizer2.zero_grad()
        actor_loss2.backward()
        self.actor_optimizer2.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local1, self.critic_target1, TAU)
        self.soft_update(self.critic_local2, self.critic_target2, TAU)
        self.soft_update(self.actor_local1, self.actor_target1, TAU)
        self.soft_update(self.actor_local2, self.actor_target2, TAU)
                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)