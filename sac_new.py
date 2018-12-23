#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:20:24 2018

@author: lihaoruo
"""
import threading
import numpy as np
import tensorflow as tf
import scipy.signal
import gym
from time import sleep

GLOBAL_STEP = 0

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class AC_Network():
    def __init__(self,s_size,a_size,scope,atrainer,ctrainer, master_net):
        if scope == 'global':
            with tf.variable_scope(scope):
                self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
                self.inputs_ = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
                self.reward = tf.placeholder(shape=[None, 1],dtype=tf.float32)
                with tf.variable_scope('actor'):
                    mu, sigma = self.build_a(self.inputs, scope='eval', trainable=True)
                    mu, sigma = mu * a_bound[1], sigma + 1e-4
                    normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), a_bound[0], a_bound[1])
                with tf.variable_scope('critic'):
                    self.value  = self.build_c(self.inputs, self.A, scope='eval', trainable=True)
                self.global_varsa = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global/actor/eval')
                self.global_varsc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global/critic/eval')
        else:
            with tf.variable_scope(scope):
                self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
                self.inputs_ = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
                self.reward = tf.placeholder(shape=[None, 1],dtype=tf.float32)
                with tf.variable_scope('actor'):
                    mu, sigma = self.build_a(self.inputs, scope='eval', trainable=True)
                    mu, sigma = mu * a_bound[1], sigma + 1e-4
                    normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), a_bound[0], a_bound[1])
                    mu_, sigma_ = self.build_a(self.inputs_, scope='target', trainable=False)
                    mu_, sigma_ = mu_ * a_bound[1], sigma_ + 1e-4
                    normal_dist_ = tf.contrib.distributions.Normal(mu_, sigma_)
                    self.A_ = tf.clip_by_value(tf.squeeze(normal_dist_.sample(1), axis=0), a_bound[0], a_bound[1])
                    
                with tf.variable_scope('critic'):
                    self.value  = self.build_c(self.inputs, self.A, scope='eval', trainable=True)
                    self.value_  = self.build_c(self.inputs_, self.A_, scope='target', trainable=False)
                
                self.memory = np.zeros((MEMORY_CAPACITY, s_size * 2 + a_size + 1), dtype=np.float32)
                self.pointer = 0
                self.local_varsa = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor/eval')
                self.local_varsc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic/eval')
                self.target_vara = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/actor/target')
                self.target_varc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/critic/target')
                
                self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                     for ta, ea, tc, ec in zip(self.target_vara, self.local_varsa, self.target_varc, self.local_varsc)]
                # actor 
                self.policy_loss = -tf.reduce_mean(self.value)
                self.gradientsa = tf.gradients(self.policy_loss,self.local_varsa)
                gradsa,self.grad_normsa = tf.clip_by_global_norm(self.gradientsa,40.0)
                self.apply_gradsa = atrainer.apply_gradients(zip(gradsa, master_net.global_varsa))

                # critic
                self.target_v = (self.reward + gamma * (self.value_ - 0.01 * normal_dist_.log_prob(self.A_)))
                self.value_loss = self.value * tf.stop_gradient(self.value - self.target_v)
                self.gradientsc = tf.gradients(self.value_loss,self.local_varsc)
                gradsc,self.grad_normsc = tf.clip_by_global_norm(self.gradientsc,40.0)
                self.apply_gradsc = ctrainer.apply_gradients(zip(gradsc, master_net.global_varsc))
                    
    
    def build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, trainable=trainable, name='l1')
            mu = tf.layers.dense(net, a_size, activation=tf.nn.tanh, trainable=trainable, name='mu')
            sigma = tf.layers.dense(net, a_size, activation=tf.nn.softplus, trainable=trainable, name='sigma')
            #return tf.multiply(a, a_bound, name='scaled_a')
            return mu, sigma
            
    def build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [s_size, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [a_size, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

class Worker():
    def __init__(self,env,name,s_size,a_size,atrainer,ctrainer,model_path, master_network):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.atrainer = atrainer
        self.ctrainer = ctrainer

        self.local_AC = AC_Network(s_size,a_size,self.name,atrainer,ctrainer, master_network)
        self.update_local_opsa = update_target_graph('global/actor/eval', self.name+'/actor/eval')
        self.update_local_opsc = update_target_graph('global/critic/eval',self.name+'/critic/eval')
        self.env = env
        
    def train(self,indices,sess,gamma):
        sess.run(self.local_AC.soft_replace)
        bt  = self.local_AC.memory[indices, :]
        bs  = bt[:, :s_size]
        ba  = bt[:, s_size:s_size+a_size]
        br  = bt[:, -s_size-1:-s_size]
        bs_ = bt[:, -s_size:]

        sess.run([self.local_AC.apply_gradsa, self.local_AC.apply_gradsc], 
                 feed_dict={self.local_AC.inputs:bs,
                            self.local_AC.reward:br,
                            self.local_AC.A:ba,
                            self.local_AC.inputs_:bs_})
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        total_steps = 0
        var = 3
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run([self.update_local_opsa,self.update_local_opsc])
                episode_reward = 0
                d = False
                s = self.env.reset()
                while total_steps< 30000:
                    #if self.local_AC.pointer > MEMORY_CAPACITY:
                    #    self.env.render()
                    a_dist = sess.run([self.local_AC.A], feed_dict={self.local_AC.inputs:[s]})[0][0]
                    a = np.clip(np.random.normal(a_dist, var), -2, 2)
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = s1
                    else:
                        s1 = s

                    self.local_AC.store_transition(s, a, r / 10, s1)
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    
                    if total_steps % 1 == 0 and d != True and self.local_AC.pointer > MEMORY_CAPACITY:
                        var *= 0.9995
                        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
                        self.train(indices,sess,gamma)
                        sess.run([self.update_local_opsa,self.update_local_opsc])
                    if d == True:
                        print 'name', self.name, 'step',total_steps, 'reward', episode_reward, 'var', var
                        break

gamma = .99 
load_model = False
model_path = './a3model'

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
tf.reset_default_graph()

atrainer = tf.train.AdamOptimizer(learning_rate=0.001)
ctrainer = tf.train.AdamOptimizer(learning_rate=0.001)

master_network = AC_Network(s_size,a_size,'global',None,None,None)
num_workers = 4
workers = []

for i in range(num_workers):
    env = gym.make(ENV_NAME)
    workers.append(Worker(env,i,s_size,a_size,atrainer,ctrainer,model_path, master_network))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    