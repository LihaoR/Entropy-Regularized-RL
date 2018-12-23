# Entropy-Regularized-RL

### A3C with Proper Entropy Bounses( Also GAE).

Run it by 
```
python A3C.py
```
### soft q learning

The gaussian kernel is from haarnoja. This is for atari, there exist some problems, I will fix them soon.

### soft actor critic

After reading paper, I think sac can be almost like a3c...And because of the entropy, it's will not converge faster than a3c in my experiment.

Run it by
```
python sac.py
```

sac_new.py is ddpg style. fixed alpha

### Paper 

[Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)
[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
