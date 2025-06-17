# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
from game import Board, Game, Game_UI
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from graphics import *
from MinMax import MinMaxAI
from collections import defaultdict, deque
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)
 
 

   
class Human_UI(object):
    """
    human player
    """

    def __init__(self, win, board):
        self.player = None
        self.win = win
        self.board = board
        self.width = board.width
        self.height = board.height
        
    def set_player_ind(self, p):
        self.player = p
        
    def manPlay_mouse(self):
        p=self.win.getMouse()
        step_width = int(450 / (self.width-1))
        step_heght = int(450 / (self.height-1))
        x=round(p.getX()/step_width)
        y=round(p.getY()/step_heght)
        # print('x,y',x,y)
        if(self.board.downOk(x,y)): 
            
            return [self.height-1-y, x]
        else: 
            print("relocate")
            return self.manPlay_mouse()

    def get_action(self, board):
        try:
            location = self.manPlay_mouse()
            # print('lcoa',location)
            move = board.location_to_move(location)
        except Exception as e:
            print(e)
            move = -1  
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def AI_compete():
    n = 5
    width, height = 9, 9
    # model_file = 'best_policy_8_8_5.model'
    model_file = 'wrlminmaxcurrent_policy.model'
    model_file2 = '12_25worlminmaxbest_policy.model'
    try:
     win = GraphWin("五子棋",570,471)
     board = Board(width=width, height=height, n_in_row=n)
     game = Game_UI(board, win)
     policy1 = PolicyValueNet(width, height, model_file = model_file,use_gpu=True)
     mcts_player1 = MCTSPlayer(policy1.policy_value_fn, c_puct=5, n_playout=5000)
     minmax_player = MinMaxAI(width, height, n_in_row=n)
     policy2 = PolicyValueNet(width, height, model_file = model_file2,use_gpu=True)
     mcts_player2 = MCTSPlayer(policy2.policy_value_fn, c_puct=5, n_playout=3000)
     game.start_play(mcts_player1,minmax_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
         print('\n\rquit')

def AI_compete2():
    n_games = 20
    n = 5
    width, height = 9, 9
    # model_file = 'best_policy_8_8_5.model'
    model_file = '12_23_1650_best_policy.model'
    model_file2 = '12_25worlminmaxbest_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        policy1 = PolicyValueNet(width, height, model_file = model_file,use_gpu=True)
        mcts_player1 = MCTSPlayer(policy1.policy_value_fn, c_puct=5, n_playout=1000)
        #  minmax_player = MinMaxAI(width, height, n_in_row=n)
        policy2 = PolicyValueNet(width, height, model_file = model_file2,use_gpu=True)
        mcts_player2 = MCTSPlayer(policy2.policy_value_fn, c_puct=5, n_playout=1000)

        print(f"start {n_games} games between {model_file} and {model_file2}")
        win_cnt = defaultdict(int)
        for i in range(int(n_games/2)):
            winner = game.start_play(mcts_player1,
                                            mcts_player2,
                                            start_player=0,
                                            is_shown=0)
            win_cnt[winner] += 1

        print("start_player: {}, {} win: {}, lose: {}, tie:{}".format(
                str(mcts_player1),
                str(mcts_player1),
                win_cnt[1], win_cnt[2], win_cnt[-1]))
            
        for i in range(int(n_games/2)):
            winner = game.start_play(mcts_player1,
                                            mcts_player2,
                                            start_player=1,
                                            is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("start_player: {}, {} win: {}, lose: {}, tie:{}".format(
                str(mcts_player2),
                str(mcts_player1),
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        print("{} vs {}, the win_ratio of {} is: {}".format(str(mcts_player1), str(mcts_player2), str(mcts_player1), win_ratio))
    except KeyboardInterrupt:
         print('\n\rquit')

def run():
    n = 5
    width, height = 9, 9
    # model_file = 'best_policy_8_8_5.model'
    model_file = 'wrlminmaxbest_policy.model'
    try:
        win = GraphWin("五子棋",570,471)
        board = Board(width=width, height=height, n_in_row=n)
        game = Game_UI(board, win)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyValueNet(width, height, model_file = model_file,use_gpu=True)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=1500)

        # human player, input your move in the format: 2,3
        human = Human_UI(win, board)

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
