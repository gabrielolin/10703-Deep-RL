#! python3

import argparse
import os

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device
        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, 1)  # Output a scalar value
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.actor_optimizer = optim.Adam(
                self.actor.parameters(),
                self.lr_actor
        )

        self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                self.lr_critic
        )

        self.critic_loss = nn.MSELoss()

        self.actor.to(self.device)
        self.critic.to(self.device)
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))
    
    def get_dist(self, state):
        logits = self.actor(state)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        if stochastic:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        # END STUDENT SOLUTION
        return action.item() 

    def calculate_n_step_bootstrap(self, states_tensor, rewards_tensor, time_step):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        T = rewards_tensor.size(0)
        end = min(time_step + self.n, T)
        with torch.no_grad():
            V_end = self.critic(states_tensor[end]).squeeze(-1) if end < T else rewards_tensor.new_zeros(())
        running_return= V_end # bootstrap once at segment end
        for t in range(end-1, time_step-1, -1):  # single reverse loop per segment
            running_return = rewards_tensor[t] + self.gamma * running_return
        # END STUDENT SOLUTION
        return running_return

    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        # Convert lists to tensors
        states = torch.cat(states, dim=0).to(self.device)      # shape: [T, state_dim]
        actions = torch.cat(actions, dim=0).to(self.device)    # shape: [T]
        rewards = torch.cat(rewards, dim=0).to(self.device)    # shape: [T]


        T = len(rewards)
        G = torch.zeros(T, device=self.device)
        if self.mode == "REINFORCE" or self.mode == "REINFORCE_WITH_BASELINE":
            running_return = 0
            for t in reversed(range(T)):
                running_return = rewards[t] + self.gamma * running_return
                G[t] = running_return
        elif self.mode == "A2C":
            for t in range(T):
                G[t] = self.calculate_n_step_bootstrap(states, rewards, t)


        actions_dist = self.get_dist(states)
        log_prob = actions_dist.log_prob(actions)

        if self.mode == "REINFORCE":
            loss = -(G * log_prob).mean()
        elif self.mode == "REINFORCE_WITH_BASELINE" or self.mode == "A2C":
            values = self.critic(states).squeeze(-1)
            advantages = G - values.detach()
            loss = -(advantages * log_prob).mean()
            # Update critic
            value_loss = self.critic_loss(values, G)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        # END STUDENT SOLUTION
        return loss.item()


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            # Collect episode data
            states = []
            actions = []
            rewards = []

            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.get_action(state_tensor, stochastic=train)
                next_state, reward, done, _, _ = env.step(action)

                # Store transition
                states.append(state_tensor)
                actions.append(torch.tensor([action]).to(self.device))
                rewards.append(torch.tensor([reward]).to(self.device))

                episode_reward += reward
                if done:
                    break
                state = next_state

            total_rewards.append(episode_reward)

            if train:
                # After episode, train with full trajectory
                loss = self.train(states, actions, rewards)

        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name,
    agent,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    # BEGIN STUDENT SOLUTION
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    graphs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graphs')
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    
    # Load D matrix for this agent
    filename_suffix = f"{agent.mode}" + (f"_n{agent.n}" if agent.n != 0 else "")
    D = np.load(os.path.join(data_path, f'D_matrix_{filename_suffix}_final.npy'))
    
    # Extract statistics across trials
    average_total_rewards = np.mean(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    
    # Create individual plot for this agent
    plt.figure()
    plt.plot(xs, average_total_rewards, label=graph_name)
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    
    plt.ylim(-max_steps * 0.01, max_steps * 1.1)
    plt.title(graph_name, fontsize=10)
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    
    # Save individual graph
    plt.savefig(os.path.join(graphs_path, f"{graph_name}.png"))
    plt.close()
    print(f"Saved graph: {graph_name}")
    # END STUDENT SOLUTION
    
    print(f"Finished: {graph_name}")


def load_saved_agents(data_path, env_name):
    """Load all saved agents from D matrix files."""
    import glob
    
    d_matrix_files = glob.glob(os.path.join(data_path, 'D_matrix_*_final.npy'))
    agents = []
    env = gym.make(env_name)
    
    for d_file in d_matrix_files:
        # Extract mode and n from filename
        filename = os.path.basename(d_file)
        # Remove 'D_matrix_' prefix and '_final.npy' suffix
        agent_info = filename[9:-10]  
        
        if '_n' in agent_info:
            mode_part, n_part = agent_info.rsplit('_n', 1)
            mode = mode_part
            n = int(n_part)
        else:
            mode = agent_info
            n = 0
        
        # Create and load agent
        agent = PolicyGradient(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            mode=mode,
            n=n,
            device="cpu",
        )
        
        # Load the trained weights from trial 0
        agent_file = os.path.join(data_path, f'agent_{agent_info}_trial_0.pth')
        if os.path.exists(agent_file):
            agent.load_state_dict(torch.load(agent_file))
            print(f"Loaded agent: {mode}" + (f" (n={n})" if n != 0 else ""))
            agents.append(agent)
    
    return agents


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    parser.add_argument(
        "--graph_only", action="store_true", help="Skip training and only generate graphs from saved data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    ### CREATE DIRECTORY FOR LOGGING
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # If graph_only flag is set, skip training and just generate graphs
    if args.graph_only:
        # Create graphs directory if it doesn't exist
        graphs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)

        # Load all saved agents
        agents = load_saved_agents(data_path, args.env_name)
        
        if agents:
            for agent in agents:
                agent_name = agent.mode + (f"_n{agent.n}" if agent.n != 0 else "")
                graph_agents(
                    graph_name=agent_name,
                    agent=agent,
                    env=None,  # Not needed for graphing
                    max_steps=args.max_steps,
                    num_episodes=args.num_episodes,
                    num_test_episodes=args.num_test_episodes,
                    graph_every=args.graph_every,
                )
        else:
            print("No saved agents found!")
        return

    env = gym.make(args.env_name)

    num_trials = 5

    num_frozen = args.num_episodes // args.graph_every
    training_episodes = args.graph_every
    testing_episodes = args.num_test_episodes

    #Log matrix for average rewards
    D = np.zeros((num_trials, num_frozen))

    for trial in range(num_trials):
        print(f"Trial {trial+1} of {num_trials}")
        #reset with random seed
        seed = 1000 + trial
        env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        #intialize policy gradient agent
        PG = PolicyGradient(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            mode=args.mode,
            n=args.n,
            device="cpu",
        )

        for i in range(num_frozen):
            print(f"Training episodes {i*training_episodes} to {(i+1)*training_episodes}")
            _ = PG.run(env, args.max_steps, training_episodes, train=True)
            print(f"Testing at episode {(i+1)*training_episodes}")
            test_rewards = PG.run(env, args.max_steps, testing_episodes, train=False)
            D[trial, i] = np.mean(test_rewards)
            print(f"Episode {(i+1)*training_episodes}: Average test reward = {np.mean(test_rewards):.2f}")

        # Save D matrix after each trial
        filename_suffix = f"{args.mode}" + (f"_n{args.n}" if args.n != 0 else "")
        np.save(os.path.join(data_path, f'D_matrix_{filename_suffix}_trial_{trial}.npy'), D)
        print(f"Saved D matrix after trial {trial+1}")

        # Save the trained agent
        torch.save(PG.state_dict(), os.path.join(data_path, f'agent_{filename_suffix}_trial_{trial}.pth'))
        print(f"Saved agent after trial {trial+1}")

    # Save final D matrix
    filename_suffix = f"{args.mode}" + (f"_n{args.n}" if args.n != 0 else "")
    np.save(os.path.join(data_path, f'D_matrix_{filename_suffix}_final.npy'), D)
    print("Saved final D matrix")

    # Create graphs directory if it doesn't exist
    graphs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../graphs')
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)

    graph_agents(
        graph_name=f"{args.mode}" + (f"_n{args.n}" if args.n != 0 else ""),
        agent=PG,
        env=env,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        num_test_episodes=args.num_test_episodes,
        graph_every=args.graph_every,
    )

    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
