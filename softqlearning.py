#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 23:04:55 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
from gaussian_kernel import adaptive_isotropic_gaussian_kernel

GLOBAL_STEP = 0
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class SoftQ_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.kernel = adaptive_isotropic_gaussian_kernel
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=self.imageIn,num_outputs=32,
                    kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=self.conv1,num_outputs=64,
                    kernel_size=[4,4],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                    inputs=self.conv2,num_outputs=64,
                    kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv3),512,activation_fn=tf.nn.relu)

            self.policy = slim.fully_connected(hidden,a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)
            self.q = slim.fully_connected(hidden,a_size,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)

            if scope != 'global':
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                #self.rewards = tf.placeholder(shape=[None],dtype=tf.float32)

                #  td error update
                q_a = self.q / tau
                self.v_next = tf.reduce_logsumexp(q_a,axis=1)
                self.q_target = tf.placeholder(shape=[None],dtype=tf.float32)
                self.readout_action = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1)
                self.td_loss = tf.reduce_mean(0.5 * tf.square(self.q_target - self.readout_action))
                
                #  svgd update
                self.ai = tf.placeholder(shape=[None,k],dtype=tf.int32)
                self.aj = tf.placeholder(shape=[None,k],dtype=tf.int32)
                self.ai_onehot = tf.one_hot(self.ai, a_size, dtype=tf.float32)
                self.aj_onehot = tf.one_hot(self.aj, a_size, dtype=tf.float32)
                self.readout_actionj = self.aj_onehot * tf.expand_dims(self.policy, axis=1)
                self.readout_actioni = self.ai_onehot * tf.expand_dims(self.policy, axis=1)
                
                self.Q_soft = tf.expand_dims(self.q, axis=1) * self.readout_actioni
                Q_soft_grad = tf.gradients(self.Q_soft, self.readout_actioni)[0]
                self.Q_soft_grad = tf.expand_dims(Q_soft_grad, axis=2)
                self.Q_soft_grad = tf.stop_gradient(self.Q_soft_grad)
                
                self.readout_actioni = tf.stop_gradient(self.readout_actioni)
                self.kernel, self.kernel_grad = self.kernel(self.readout_actioni, self.readout_actionj)
                self.kernel = tf.expand_dims(self.kernel, axis=3)
                
                self.action_gradient = tf.reduce_mean(self.kernel * self.Q_soft_grad + self.kernel_grad, axis=1)
                self.action_gradients = tf.stop_gradient(self.action_gradient)
                
                self.su = tf.reduce_sum(tf.reduce_sum(self.action_gradients * self.readout_actionj, axis=2), axis=1)
                self.surrogate_loss = - tf.reduce_mean(self.su)
                
                #action_gradients = tf.reduce_mean(self.kernel * self.Q_soft_grad + self.kernel_grad, axis=1)
                #self.gradient = tf.gradients(self.readout_actionj, local_vars, grad_ys=action_gradients)
                #self.surrogate_loss = -tf.reduce_sum(local_vars * tf.stop_gradient(self.gradient))
                
                # total loss
                self.loss = self.td_loss + self.surrogate_loss
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.local_softq = SoftQ_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        
    def train(self,rollout,sess,gamma,s1):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        #dones = rollout[:,4]
        ai = rollout[:,5]
        aj = rollout[:,6]
    
        #print ai, np.shape(ai)
        #print np.vstack(ai), np.shape(np.vstack(ai))
        v_next = sess.run(self.local_softq.v_next, 
                          feed_dict={self.local_softq.inputs:np.stack(next_observations)})
        q_target = gamma * v_next + rewards - 1.0
        #print 'q_target', np.mean(q_target)
        #print v_target, np.shape(v_target)
        #actions_next = np.argmax(q_target, axis=1)
        #k, kg = sess.run([self.local_softq.kernel, self.local_softq.kernel_grad],
        #                 feed_dict={self.local_softq.inputs:np.vstack(observations),
        #                            self.local_softq.ai:np.vstack(ai), 
        #                            self.local_softq.aj:np.vstack(aj)})
        #print np.shape(k)
        #print np.shape(kg)
        
        feed_dict = {self.local_softq.q_target:q_target,
                     self.local_softq.inputs:np.vstack(observations),
                     self.local_softq.actions:actions,
                     #self.local_softq.rewards:rewards,
                     self.local_softq.ai:np.vstack(ai),
                     self.local_softq.aj:np.vstack(aj)}
        
        loss,td_loss,read,q,qg,_ = sess.run([self.local_softq.loss,
                                           self.local_softq.td_loss,
                                           #self.local_softq.surrogate_loss,
                                           self.local_softq.readout_action,
                                           self.local_softq.Q_soft,
                                           self.local_softq.Q_soft_grad,
                                           #self.local_softq.action_gradients,
                                           #self.local_softq.gradient,
                                           self.local_softq.apply_grads],
                                           feed_dict=feed_dict)
        #print 'gradient',qg,np.shape(qg)
        #print 'read    ', read
        print q_target
        return loss, td_loss, read, q_target
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                s = process_frame(s)
                while not d:
                    GLOBAL_STEP += 1
                    a_dist = sess.run(self.local_softq.policy, feed_dict={self.local_softq.inputs:[s]})[0]
                    a = np.random.choice(a_dist, p=a_dist)
                    a = np.argmax(a_dist==a)
                    ai, aj = np.zeros([k]), np.zeros([k])
                    for i in range(k):
                        b = np.random.choice(a_dist, p=a_dist)
                        b = np.argmax(a_dist==b)
                        c = np.random.choice(a_dist, p=a_dist)
                        c = np.argmax(a_dist==c)
                        ai[i] = b
                        aj[i] = c
                        
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,ai,aj])
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    if len(episode_buffer) == batch_size and d != True:
                        loss,td_loss,su_loss,q_target = self.train(episode_buffer,sess,gamma,s1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)

                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                    print ('td_loss:',td_loss,'su_loss',su_loss,'q_target', np.mean(q_target))
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99
s_size = 7056
load_model = False
model_path = './model'
batch_size = 2
tau = 1

benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
master_network = SoftQ_Network(s_size,a_size,'global',None)
num_workers = 1
workers = []

### soft dqn hyperparameters ###
k = 4
################################
for i in range(num_workers):
    env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
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


