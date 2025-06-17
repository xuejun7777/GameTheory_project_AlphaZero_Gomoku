## AlphaZero-Gomoku

References:  
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. AlphaGo Zero: Mastering the game of Go without human knowledge



### Requirements
- Python 
- Numpy 
- PyTorch >= 0.2.0    


### Getting Started
运行以下命令训练或者评估（与pure_mcts对弈）
```
python human_play.py  
```

运行human_play.py可以在UI界面中与MCTS_player（训练的模型）对弈（run函数），或者运行AI_compete在UI界面中看minmax算法与MCTS_player对弈以及两个模型权重不同的MCTS_player对弈（model_file是模型权重名字），运行AI_compete2函数可以无需UI界面评估两个模型或者模型与minmax算法的胜负情况。
```
python human_play.py  
```
运行minmax_ui.py可以在UI界面中与minmax算法对弈。
```
python minmax_UI.py  
```


This work is highly based on https://github.com/junxiaosong/AlphaZero_Gomoku.git, thanks to authors.
