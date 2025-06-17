# minmax_ai.py
import time
import numpy as np
class MinMaxAI:
    def __init__(self, width=16, height=16, n_in_row=5,player=1):
        self.width = width
        self.height = height
        self.num = np.array([[0 for _ in range(width)] for _ in range(height)])
        self.dx = [1, 1, 0, -1, -1, -1, 0, 1]  # x,y方向向量
        self.dy = [0, 1, 1, 1, 0, -1, -1, -1]
        self.is_end = False
        self.player = player  # AI下棋标志
        self.L1_max = -100000  # 剪枝阈值
        self.L2_min = 100000
        self.move=-1
        self.move_probs=np.zeros(self.width*self.height)
        
        
       
    def in_board(self, x, y):
        return 0 <= x < len(self.num) and 0 <= y < len(self.num)
        
    def is_empty(self):
         if((np.any(np.equal(self.num,2))) or (np.any(np.equal(self.num,1)))):
            return False
         else:
            return True
    def game_over(self, x, y):
        for u in range(4):
            if (self.num_inline(x, y, u) + self.num_inline(x, y, u + 4)) >= 4:
                self.is_end = True
                return True
        return False     
    def down_ok(self, x, y):
        return self.in_board(x, y) and self.num[x][y] == 0

    def same_color(self, x, y, i):
        return self.in_board(x, y) and self.num[x][y] == i

    def num_inline(self, x, y, v):
        i = x + self.dx[v]
        j = y + self.dy[v]
        s = 0
        ref = self.num[x][y]
        if ref == 0:
            return 0
        while self.same_color(i, j, ref):
            s += 1
            i += self.dx[v]
            j += self.dy[v]
        return s

    def live_four(self, x, y):
        key = self.num[x][y]
        s = 0
        for u in range(4):
            samekey = 1
            samekey, i = self.num_of_same_key(x, y, u, 1, key, samekey)
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            samekey, i = self.num_of_same_key(x, y, u, -1, key, samekey)
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            if samekey == 4:
                s += 1
        return s

    def chong_four(self, x, y):
        key = self.num[x][y]
        s = 0
        for u in range(8):
            samekey = 0
            flag = True
            i = 1
            while self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key) or flag:
                if not self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key):
                    if flag and self.in_board(x + self.dx[u] * i, y + self.dy[u] * i) and self.num[x + self.dx[u] * i][y + self.dy[u] * i] != 0:
                        samekey -= 10
                    flag = False
                samekey += 1
                i += 1
            i -= 1
            if not self.in_board(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            samekey, i = self.num_of_same_key(x, y, u, -1, key, samekey)
            if samekey == 4:
                s += 1
        return s - self.live_four(x, y) * 2

    def live_three(self, x, y):
        key = self.num[x][y]
        s = 0
        for u in range(4):
            samekey = 1
            samekey, i = self.num_of_same_key(x, y, u, 1, key, samekey)
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            if not self.down_ok(x + self.dx[u] * (i + 1), y + self.dy[u] * (i + 1)):
                continue
            samekey, i = self.num_of_same_key(x, y, u, -1, key, samekey)
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            if not self.down_ok(x + self.dx[u] * (i - 1), y + self.dy[u] * (i - 1)):
                continue
            if samekey == 3:
                s += 1
        for u in range(8):
            samekey = 0
            flag = True
            i = 1
            while self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key) or flag:
                if not self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key):
                    if flag and self.in_board(x + self.dx[u] * i, y + self.dy[u] * i) and self.num[x + self.dx[u] * i][y + self.dy[u] * i] != 0:
                        samekey -= 10
                    flag = False
                samekey += 1
                i += 1
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            if self.in_board(x + self.dx[u] * (i - 1), y + self.dy[u] * (i - 1)) and self.num[x + self.dx[u] * (i - 1)][y + self.dy[u] * (i - 1)] == 0:
                continue
            samekey, i = self.num_of_same_key(x, y, u, 1, key, samekey)
            if not self.down_ok(x + self.dx[u] * i, y + self.dy[u] * i):
                continue
            if samekey == 3:
                s += 1
        return s
    def set_player_ind(self, p):
        self.player = p
    def over_line(self, x, y):
        flag = False
        for u in range(4):
            if (self.num_inline(x, y, u) + self.num_inline(x, y, u + 4)) > 4:
                flag = True
        return flag

    def ban(self, x, y):
        if self.same_color(x, y, 3 - self.player):
            return False
        flag = (self.live_three(x, y) > 1 or self.over_line(x, y) or (self.live_four(x, y) + self.chong_four(x, y)) > 1)
        return flag

    def num_of_same_key(self, x, y, u, i, key, sk):
        if i == 1:
            while self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key):
                sk += 1
                i += 1
        elif i == -1:
            while self.same_color(x + self.dx[u] * i, y + self.dy[u] * i, key):
                sk += 1
                i -= 1
        return sk, i

   

    def get_score(self, x, y):
        if self.ban(x, y):
            return 0
        if self.game_over(x, y):
            self.is_end = False
            return 10000
        score = self.live_four(x, y) * 1000 + (self.chong_four(x, y) + self.live_three(x, y)) * 100
        for u in range(8):
            if self.in_board(x + self.dx[u], y + self.dy[u]) and self.num[x + self.dx[u]][y + self.dy[u]] != 0:
                score += 1
        return score

    def AI1(self,X,Y):
        self.L1_max = -100000
        if self.is_empty():
            return int(self.width/2),int(self.height/2)
        keyi = -1;
        keyj = -1;
        for x,y in zip(X,Y):
                if (not self.down_ok(x, y)):
                    continue
                self.num[x][y] = self.player
                tempp = self.get_score(x, y)
                if tempp == 0:
                    self.num[x][y] = 0;
                    continue
                if tempp == 10000:
                    return x, y
                tempp = self.AI2(X,Y)
                self.num[x][y] = 0
                if tempp > self.L1_max:  # 取极大
                    self.L1_max = tempp;
                    keyi = x;
                    keyj = y
        #print(self.L1_max)
        return keyi, keyj

    def AI2(self,X,Y):
        self.L2_min = 100000
        for x,y in zip(X,Y):
                if not self.down_ok(x, y):
                    continue
                self.num[x][y] = 3 - self.player
                tempp = self.get_score(x, y)
                if tempp == 0:
                    self.num[x][y] = 0;
                    continue
                if tempp == 10000:
                    self.num[x][y] = 0;
                    return -10000
                tempp = self.AI3(tempp,X,Y)
                if tempp < self.L1_max:  # L1层剪枝
                    self.num[x][y] = 0;
                    return -10000
                self.num[x][y] = 0
                if tempp < self.L2_min:  # 取极小
                    self.L2_min = tempp
        return self.L2_min
    
    def AI3(self, p2,X,Y):
        keyp = -100000
        for x,y in zip(X,Y):
                if not self.down_ok(x, y):
                    continue
                self.num[x][y] = self.player
                tempp = self.get_score(x, y)
                if tempp == 0:
                    self.num[x][y] = 0;
                    continue
                if tempp == 10000:
                    self.num[x][y] = 0;
                    return 10000
                if tempp - p2 * 2 > self.L2_min:  # L2层剪枝
                    self.num[x][y] = 0;
                    return 10000
                self.num[x][y] = 0
                if tempp - p2 * 2 > keyp:  # 取极大
                    keyp = tempp - p2 * 2
        return keyp

    def __str__(self):
        return "MinMax"

    def get_action(self,board,return_prob=0):
        
        self.num=self.num*0
        self.move_probs=np.zeros(board.width*board.height)
        for key,value in board.states.items():
            h=key//self.height
            w=key%self.width
            self.num[w][self.height-1-h]=value
            
        self.player=board.current_player
        X=np.array(board.availables)%self.width
        Y=self.height-1-np.array(board.availables)//self.height
        x,y=self.AI1(X,Y)
        if x == -1 and y == -1:
            random_index = np.random.choice(len(board.availables))
            x = X[random_index]
            y = Y[random_index]
        h,w=self.height-1-y,x
        self.move= h * self.width + w
        # print("minmax",x,y,h,w,self.move)
        self.move_probs[self.move]=1
        if return_prob:
            return self.move,self.move_probs
        else:
            return self.move