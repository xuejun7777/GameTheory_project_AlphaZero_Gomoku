# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(256, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(256, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.rl_coff = 0.2
        # the policy value net module
        if self.use_gpu:
            self.device = torch.device("cuda")
            self.policy_value_net = Net(board_width, board_height).to(self.device)
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = np.array(state_batch)
            state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).detach().cpu().numpy()
            return act_probs, value.detach().cpu().numpy()

        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            current_state = torch.tensor(current_state, dtype=torch.float32).to(self.device)
            log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).detach().cpu().numpy().flatten()
            value = value.detach().cpu().numpy()[0][0]

        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr, episode_data=None):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = np.array(state_batch)
            mcts_probs = np.array(mcts_probs)
            winner_batch = np.array(winner_batch)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
            winner_batch = torch.FloatTensor(winner_batch).to(self.device)
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        #Reinforce algorithm as in alphago2016
        if episode_data is not None:
            # if np.random.rand() < 0.5:
            #     episode_data = [data for data in episode_data if data[2]==1.0 or data[2]==0.0]
            # else:
            #     episode_data = [data for data in episode_data if data[2]==-1.0 or data[2]==0.0]
            # episode_data = [(data[0], data[1], -0.25) if data[2] == 0.0 else data for data in episode_data]
            # episode_data = [(data[0], data[1], -0.25) if data[2] == 0.0 else data for data in episode_data if (np.random.rand() < 0.5 and data[2] == 1.0 or data[2] == 0.0) or (np.random.rand() >= 0.5 and data[2] == -1.0 or data[2] == 0.0)]
            # print("do rl update")
            debug_list = [episode_data[0][2],episode_data[1][2]]
            allowed_values = {0.0, 1.0} if np.random.rand() < 0.5 else {0.0, -1.0}
            episode_data = [
                (data[0], data[1], -0.25 if data[2] == 0.0 else data[2])
                for data in episode_data
                if data[2] in allowed_values
            ]
            episode_state = torch.FloatTensor(np.array([data[0] for data in episode_data])).to(self.device)
            episode_act_probs = torch.FloatTensor(np.array([data[1] for data in episode_data])).to(self.device)
            episode_returns = torch.FloatTensor(np.array([data[2] for data in episode_data])).to(self.device)
            # episode_act = torch.stack([torch.distributions.Categorical(probs).sample() for probs in episode_act_probs]).to(self.device)
            try:
                episode_act = torch.stack([torch.distributions.Categorical(probs).sample() for probs in episode_act_probs]).to(self.device)
            except RuntimeError as e:
                print("Error: winner_value", e, debug_list)
            log_probs4rl, _ = self.policy_value_net(episode_state)
            # print(mcts_probs.shape, log_act_probs.shape,(mcts_probs*log_act_probs).shape)
            log_probs = log_probs4rl.gather(1, episode_act.view(-1,1)).squeeze()
            # print(episode_state.shape, log_probs.shape, episode_returns.shape)
            rl_policy_loss = -(log_probs*episode_returns).mean() * self.rl_coff

            loss += rl_policy_loss
            
        
        # backward and optimize
        loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad.clip_grad_norm_(
        #     self.policy_value_net.parameters(),
        #     max_norm=1.0
        # )
        
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
