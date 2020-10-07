#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:18:32 2020

@author: Shihab
"""

import gym
import numpy as np
import tensorflow as tf


def no_bg(image):               #Eliminates the background of the image before processing
    image[image == 109] = 0
    image[image == 144] = 0
    return image

def halved(image):              #Splits the image by half/useful for cropping
    return image[::2, ::2, :]

def no_colors(image):           #Eliminates any irrelevant or unnecessary colors for the board game
    return image[:, :, 0]

def preprocess_obs(input_dm, input_obs, prev_obs):  #The preprocessing phase of the image
    processed_obs = input_obs[35:195]               #Cropped image
    processed_obs = halved(processed_obs)
    processed_obs = no_bg(processed_obs)
    processed_obs = no_colors(processed_obs)
    processed_obs[processed_obs != 0] = 1
    processed_obs = processed_obs.astype(np.float).ravel()
    if prev_obs is None:
        input_obs = np.zeros(input_dm)
    else:
        value = processed_obs - prev_obs
        input_obs = value
    prev_obs = processed_obs
    return input_obs, prev_obs


def ReLU(v):                    #Most useful activation function since it does not activate all
    v[v < 0] = 0                #the neurons at the same time
    return v

def n_net(obs_matrix, weights):                 #Our neural network function which takes in our observation matrix
    values1 = np.dot(weights['1'], obs_matrix)  #to calculate the output nodes
    values1 = ReLU(values1)
    values2 = np.dot(values1, weights['2'])
    num = np.exp(values2 * -1)
    num += 1.0
    num = 1.0/num
    values2 = num
    return values1, values2

def Up_or_Down(prob):                   #Our function that determines whether the next action is up or down
    next_move = 0                       #movement, utilizes the random method to compare with probability
    if np.random.uniform() >= prob:     # 3 represents down in OpenGym and 2 represents up
        next_move = 3
    else:
        next_move = 2
    return next_move

def gradient(grad, values, obs_vals, weights):  #Computes the gradient by using derivatives of the weights
    vals1 = np.outer(grad, weights['2'])
    vals1 = ReLU(vals1)
    vals1_trans = vals1.T
    deriv1 = np.dot(vals1_trans, obs_vals)
    vals2_trans = values.T
    deriv2 = np.dot(vals2_trans, grad).ravel()
    
    res = {
        '1': deriv1,
        '2': deriv2
    }
    
    return res

def weights_change(weights, lr, dr, expected_g, g_dic): #Changes and updates the weights based on the learning rate,
    for weight in weights.keys():                       #weights, and a couple of other parameters.
        g = g_dic[weight]
        temp = (1-dr) * (g**2)
        val = dr * expected_g[weight]
        val += temp
        expected_g[weight] = val
        temp = expected_g[weight] + (1e-5)
        temp = np.sqrt(temp)
        val = lr * g
        val /= temp
        weights[weight] += val
        g_dic[weight] = np.zeros_like(weights[weight])



def lower_rewards(reward, g):                   #This function priortizes recent actions or moves over those that
    rewards = np.zeros_like(reward)             #occurs 15 to 20 moves before the final result. Most recent move
    update = 0                                  # will usually dictate the AI reward and future behavior
    for r in reversed(range(0, reward.size)):
        if reward[r] != 0:
            update = 0
        update = update * g 
        update += reward[r]
        rewards[r] = update
    return rewards

def grad_rewards(grad, epi_reward, g):          #This function tries to normalize the rewards by utilizing the
    epi_rewards = lower_rewards(epi_reward, g)  #gradient
    average = np.mean(epi_rewards)
    stan_dev = np.std(epi_rewards)
    epi_rewards -= average
    epi_rewards /= stan_dev
    res = grad * epi_rewards
    return res



def train():     #Our main function that creates the environment and uses the methods above to create the episodes
    env = gym.make("Pong-v0")   #Imports the game Pong from OpenGym
    input_obs = env.reset() # Makes the game

    # parameters for training
    epi_number = 0      #Current episode/game number, starts from 0
    input_dm = 80 * 80  #Size/inner dimensions of Pong board/image
    batch_size = 10     #Fixed set of episodes/games that determine where the AI moves towards after 10 games are over
    g = 0.95            #Downgraded the discount factor gamma from 0.99 to 0.95
    dr = 0.99           #Decay rate
    lr = 1e-4           #learning rate
    num_neurons = 200   #Number of nodes/neurons
    reward_gain = 0     #reward total, start at 0, resets after every episode
    update = None
    prev_obs = None

    weights = {
        '1': np.random.randn(num_neurons, input_dm) / np.sqrt(input_dm),
        '2': np.random.randn(num_neurons) / np.sqrt(num_neurons)
    }

    #rmsprop algorithm arguments
    expected_g = {}
    g_dic = {}
    for weight in weights.keys():
        expected_g[weight] = np.zeros_like(weights[weight])
        g_dic[weight] = np.zeros_like(weights[weight])

    epi_values, epi_obs, epi_grad, epi_rewards = [], [], [], []


    while True:
        env.render()
        processed_obs, prev_obs = preprocess_obs(input_dm, input_obs, prev_obs)
        values, down_prob = n_net(processed_obs, weights)
    
        epi_obs.append(processed_obs)
        epi_values.append(values)

        move = Up_or_Down(down_prob)            #Returns the action/move

        input_obs, reward, done, info = env.step(move) #Produce the move in the game

        reward_gain += reward
        epi_rewards.append(reward)

        if move == 3:       #Move is down, reward them
            grad_loss = 1
        else:
            grad_loss = 0   #Move is up, no reward
        grad_losses = grad_loss - down_prob
        epi_grad.append(grad_losses)


        if done: #One game/episode is over
            epi_number += 1 #Increment for next game

            #Place and add the current values in the array for the current game
            epi_rewards = np.vstack(epi_rewards)
            epi_grad = np.vstack(epi_grad)
            epi_values = np.vstack(epi_values)
            epi_obs = np.vstack(epi_obs)

            epi_dis_grad = grad_rewards(epi_grad, epi_rewards, g)
            grad = gradient(epi_dis_grad, epi_values, epi_obs, weights)

            #Add up all gradients for once we reach ten games, then AI will move in that direction
            for weight in grad:
                value = grad[weight]
                g_dic[weight] += value

            if epi_number % batch_size == 0:
                weights_change(weights, lr, dr, expected_g, g_dic)

            epi_values, epi_obs, epi_grad, epi_rewards = [], [], [], [] # reset the arrays
            input_obs = env.reset() # reset the environment
            if update is None:
                update = reward_gain
            else:
                update = update * 0.99 + reward_gain * 0.03 #Upgraded the reward for achieving a hit
            reward_gain = 0
            prev_obs = None

train()