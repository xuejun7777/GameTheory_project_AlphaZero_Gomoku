# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from graphics import *
from MinMax import MinMaxAI

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
    
    def inBoard(self, x,y):
        if(x>=0 and x<=self.width-1 and y>=0 and y<=self.height-1): return True
        else: return False
        
    def downOk(self, x,y):
        if(self.inBoard(x,y)): return True
        else: return False


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self.minmax = MinMaxAI(board.width, board.height, board.n_in_row)
        self.mix_strategy = kwargs.get('mix_strategy', None)
        

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print(loc)
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3, batch_index=0):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            if self.mix_strategy is not None:
                if(np.random.rand()<self.mix_strategy and batch_index < 1200):
                    move, move_probs=self.minmax.get_action(self.board,return_prob=1)
                    player.mcts.update_with_move(move)
                else:
                    move, move_probs = player.get_action(self.board,temp=temp,return_prob=1)
            else:
                move, move_probs = player.get_action(self.board,temp=temp,return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

class Game_UI(object):
    """game server"""

    def __init__(self, board, win, **kwargs):
        self.board = board
        self.list = []
        self.win = win
        self.aiFirst = Text(Point(510,100),"")
        self.manFirst = Text(Point(510,140),"")
        self.notice = Text(Point(510,290),"") #提示轮到谁落子
        self.notice.setFill('red')
        self.last_ai = Text(Point(510,330),"") #AI最后落子点
        self.last_man = Text(Point(510,370),"") #玩家最后落子点
        self.QUIT = Text(Point(510,20),"退出")
        self.QUIT.setFill('red')
        self.RESTART = Text(Point(510,60),"重玩")
        self.RESTART.setFill('red')
        self.width = board.width
        self.height = board.height
        self.players = [1, 2]  # player1 and player2
        self.minmax = MinMaxAI(board.width, board.height)
        self.mix_strategy = kwargs.get('mix_strategy', None)

    def draw_init(self):
        for i in range(len(self.list)):
            self.list[-1].undraw()
            self.list.pop(-1)
        self.notice.setText("")
        self.last_ai.setText("")
        self.last_man.setText("")
        
    def drawWin(self):
        self.win.setBackground('yellow')
        self.step_width = int(450 / (self.width - 1))
        self.step_height = int(450 / (self.height - 1))
        for i in range(10,461,self.step_width):
            line=Line(Point(i,10),Point(i,460))
            line.draw(self.win)
        for j in range(10,461,self.step_height):
            line=Line(Point(10,j),Point(460,j))
            line.draw(self.win)
        Rectangle(Point(461,5),Point(550,35)).draw(self.win)
        Rectangle(Point(461,45),Point(550,75)).draw(self.win)
        Rectangle(Point(461,85),Point(550,115)).draw(self.win)
        Rectangle(Point(461,125),Point(550,155)).draw(self.win)
        Rectangle(Point(462,275),Point(558,305)).draw(self.win)
        Rectangle(Point(462,307),Point(558,395)).draw(self.win)
        self.aiFirst.draw(self.win)
        self.manFirst.draw(self.win)
        self.notice.draw(self.win)
        self.last_ai.draw(self.win)
        self.last_man.draw(self.win)
        self.QUIT.draw(self.win)
        self.RESTART.draw(self.win)

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print(loc)
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def go(self, x,y):
        c=Circle(Point(10+x*self.step_width,10+y*self.step_height),13)
        current_player = self.board.get_current_player()
        # print(current_player, self.start_player)
        if(self.start_player==current_player): c.setFill('black')
        else: c.setFill('white')

        c.draw(self.win)
        self.list.append(c)

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        self.draw_init()
        self.drawWin()
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        self.start_player = self.players[start_player]
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            self.notice.setText(f" {str(player_in_turn)}走棋..")
            move = player_in_turn.get_action(self.board)
            loca = self.board.move_to_location(move)
            # print(loca[1],self.height - loca[0] - 1)
            # print(move)
            # print(self.board.availables)
            self.go(loca[1],self.height - loca[0] - 1)
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner


    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            if self.mix_strategy is not None:
                if(np.random.rand()<self.mix_strategy):
                    move, move_probs=self.minmax.get_action(self.board,return_prob=1)
                else:
                    move, move_probs = player.get_action(self.board,temp=temp,return_prob=1)
            else:
                move, move_probs = player.get_action(self.board,temp=temp,return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)