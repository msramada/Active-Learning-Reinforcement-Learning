import torch
import torch.optim as optim
import torch.nn as nn
from ActorCritic import *
from utils import *

class DDPGagent:
    def __init__(self, rx , ru, device, hidden_size=12, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.95, max_memory_size=640, tau=1e-2):
        # Params
        self.num_states = rx
        self.num_actions = ru
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.tau = tau
        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.device = device
        self.actor.device = device
        self.critic.device = device
        self.actor_target.device = device
        self.critic_target.device = device
        self.critic.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.actor_target.eval()
        # Training
        self.memory = ReplayMemory(max_memory_size) 
        self.critic_criterion  = nn.MSELoss()      
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
    
    def get_action(self, state):
        action = self.actor.forward(state)
        return action
    
    def update_critic(self, batch_size):
        self.critic.train()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
    
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        #torch.stack(list(batch.state), dim= 0 ).reshape(self.num_states, batch_size).shape 
        states = torch.flatten(torch.cat(batch.state), start_dim=1).to(self.device)
        actions = torch.flatten(torch.cat(batch.action), start_dim=1).to(self.device)
        rewards = torch.tensor(batch.reward).reshape(batch_size,1).to(self.device)
        next_states = torch.flatten(torch.cat(batch.next_state), start_dim=1).to(self.device)

        # Critic loss DQN
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        self.critic.eval()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update_actor(self, batch_size):
        self.actor.train()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)    
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.flatten(torch.cat(batch.state), start_dim=1).to(self.device)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor.eval()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))