#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=memory_size)
        # END STUDENT SOLUTION
        pass


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        return random.sample(self.memory, self.batch_size)
        # END STUDENT SOLUTION
        pass


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(transition)
        # END STUDENT SOLUTION
        pass



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, double_dqn, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        self.lr_q_net = lr_q_net
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_batch_size = replay_buffer_batch_size    

        self.double_dqn = double_dqn

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size),
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_memory = ReplayMemory(self.replay_buffer_size, self.replay_buffer_batch_size)
        self.q_net = q_net_init().to(self.device)
        self.target_net = q_net_init().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr_q_net)
        
        # Counter for tracking training steps
        self.training_steps = 0
        # END STUDENT SOLUTION


    def forward(self, states, next_states, rewards, dones):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        current_q_values = self.q_net(states)
        
        if self.double_dqn:
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN: use main network for both selection and evaluation
            next_q_values = self.q_net(next_states).max(1)[0]
        
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        return current_q_values, target_q_values
        # END STUDENT SOLUTION

    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()
        # END STUDENT SOLUTION
    
    def train(self, replay_batch):
        # train the q network given the replay batch
        states, actions, rewards, next_states, dones = zip(*replay_batch)
        # make sure replay batch on device
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values, target_q_values = self.forward(states, next_states, rewards, dones)
        current = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(current, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increment training step counter
        self.training_steps += 1
        
        # Update target network every self.target_update steps
        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss

    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = self.get_action(state, stochastic=train)
                next_state, reward, done, _, _ = env.step(action)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([action]).to(self.device)
                reward_tensor = torch.tensor([reward]).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Store transition
                self.replay_memory.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done))

                episode_reward += reward

                state = next_state  

                if train:
                    #update Q net after every step
                    sample_batch = self.replay_memory.sample_batch()
                    loss = self.train(sample_batch)

                if done:
                    break

            total_rewards.append(episode_reward)

        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    average_total_rewards = np.mean(mean_undiscounted_returns, axis=0)
    min_total_rewards = np.min(mean_undiscounted_returns, axis=0)
    max_total_rewards = np.max(mean_undiscounted_returns, axis=0)

    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    env = gym.make(args.env_name)

    num_trials = args.num_runs
    num_frozen = args.num_episodes // args.test_frequency
    training_episodes = args.test_frequency
    testing_episodes = 20

    #Log matrix for average rewards
    D = np.zeros((num_trials, num_frozen))

    for trial in range(num_trials):
        print(f"Trial {trial+1} of {num_trials}")
        #reset with random seed
        env.reset(seed=0)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        action_size = env.action_space.n
        state_size = env.observation_space.shape[0] 
        #initialize DQN agent   
        dqn = DeepQNetwork(state_size=state_size, action_size=action_size, double_dqn=args.double_dqn, device='cuda' if torch.cuda.is_available() else 'cpu')

        #initialize memory with random policy
        state, _ = env.reset()
        for step in range(dqn.burn_in):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convert to tensors for storage
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(dqn.device)
            action_tensor = torch.tensor([action]).to(dqn.device)
            reward_tensor = torch.tensor([reward]).to(dqn.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(dqn.device)
            
            dqn.replay_memory.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done))
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state

        for i in range(num_frozen):
            print(f"Training episodes {i*training_episodes} to {(i+1)*training_episodes}")
            _ = dqn.run(env, args.max_steps, training_episodes, train=True)
            print(f"Testing at episode {(i+1)*training_episodes}")
            test_rewards = dqn.run(env, args.max_steps, testing_episodes, train=False)
            D[trial, i] = np.mean(test_rewards)
            print(f"Episode {(i+1)*training_episodes}: Average test reward = {np.mean(test_rewards):.2f}")

        # Save D matrix after each trial
        filename_suffix = f"{'Double_' if args.double_dqn else ''}DQN"
        np.save(os.path.join(data_path, f'D_matrix_{filename_suffix}_trial_{trial}.npy'), D)
        print(f"Saved D matrix after trial {trial+1}")

        # Save the trained agent
        torch.save(dqn.state_dict(), os.path.join(data_path, f'agent_{filename_suffix}_trial_{trial}.pth'))
        print(f"Saved agent after trial {trial+1}")

    # Save final D matrix
    filename_suffix = f"{'Double_' if args.double_dqn else ''}DQN"
    np.save(os.path.join(data_path, f'D_matrix_{filename_suffix}_final.npy'), D)
    print("Saved final D matrix")

    # Create graphs directory if it doesn't exist
    graphs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './graphs')
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)

    graph_agents(
        graph_name=f"{'Double ' if args.double_dqn else ''}DQN on {args.env_name}",
        mean_undiscounted_returns=D,
        test_frequency=args.test_frequency,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes
    )

    # END STUDENT SOLUTION


if '__main__' == __name__:
    main()

