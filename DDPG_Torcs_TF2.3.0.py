""" 
In this project, I implemented DDPG algorithms in torcs. I have used snakeoil and gymtorcs to intereact with torcs environment.
@Author: Rafail Islam
"""

from gym_torcs import TorcsEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as ks
from tensorflow.keras.layers import Input,Dense,concatenate,add
# importing classes
from ReplayBuffer import ReplayBuffer
from OUActionNoise import OUActionNoise
import json

noise = OUActionNoise()

def get_actor(HIDDEN1_NODES,HIDDEN2_NODES):
    x = Input(shape=(29,) )   
    h0 = Dense(HIDDEN1_NODES, activation='relu')(x)
    h1 = Dense(HIDDEN2_NODES, activation='relu')(h0)
    Steering = Dense(1,activation='tanh', kernel_initializer=tf.random_normal_initializer(stddev=1e-4))(h1)  
    Acceleration = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1)   
    Brake = Dense(1,activation='sigmoid', kernel_initializer=tf.random_normal_initializer(stddev=1e-4) )(h1) 
    V = concatenate([Steering,Acceleration,Brake])          
    model = ks.Model(inputs=x,outputs=V)
    
    return model

def get_critic(HIDDEN1_NODES,HIDDEN2_NODES):
    
    S = Input(shape=(29,))  
    A = Input(shape=(3,),name='action2')
    
    w1 = Dense(HIDDEN1_NODES, activation='relu')(S)
    a1 = Dense(HIDDEN2_NODES, activation='linear')(A) 
    
    h1 = Dense(HIDDEN2_NODES, activation='linear')(w1)
    h2 = add([h1,a1])    
    h3 = Dense(HIDDEN2_NODES, activation='relu')(h2)
    
    V = Dense(3,activation='linear')(h3)   
    model = ks.Model(inputs=[S,A],outputs=V)
    
    #adam = ks.optimizers.Adam(lr=self.LEARNING_RATE)
    #model.compile(loss='mse', optimizer=adam)
    return model

@tf.function
def target_values(new_states, target_actor,target_critic):
    
    # target action for batch size new_states
    target_actions = target_actor(new_states)
    
    #tf.print("target_actions",type(target_actions))
    #tf.print(type(new_states))
    
    # target Qvalue for the batch size
    target_q_values = target_critic([new_states,target_actions ])  
    return target_q_values

@tf.function
def update(actor_model,critic_model,states,actions,y,actor_optimizer,critic_optimizer):
    y = tf.cast(y, dtype=tf.float32)
    with tf.GradientTape() as tape:
        critic_value = critic_model([states,actions])
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
    
    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
    
    with tf.GradientTape() as tape:
        actions = actor_model(states)
        #tf.print("actions type",type(actions))
        critic_value = critic_model([states, actions], training=True)
        
        actor_loss = -tf.math.reduce_mean(critic_value)
        
    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))
    
    #print(type(critic_loss))
    #print(critic_loss.dtype)
    #tf.enable_eager_execution() 
    #tf.print(tf.executing_eagerly())
    #critic_loss = critic_loss.numpy()
    return actor_loss, critic_loss

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
    
def aiAgent(train_indicator):
    #disable_eager_execution()
    print(tf.executing_eagerly())
    # initializing variables
    
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)
    vision = False

    EXPLORE = 100000.
    total_episodes = 200
    total_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    hidden_unit1 = 300
    hidden_unit2 = 600
    #
    #actor = ActorModel( state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    #critic = CriticModel(state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    
    actor_model  = get_actor(hidden_unit1, hidden_unit2)
    critic_model = get_critic(hidden_unit1, hidden_unit2)
    
    #actor_model.summary()
    #critic_model.summary()
    
    target_actor = get_actor(hidden_unit1, hidden_unit2)
    target_critic = get_critic(hidden_unit1, hidden_unit2)
    
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    
    critic_optimizer = tf.keras.optimizers.Adam(LRC)
    actor_optimizer = tf.keras.optimizers.Adam(LRA)

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    
    #Now load the weight
    print("Now we load the weight")
    try:
        actor_model.load_weights("actormodel_1.h5")
        critic_model.load_weights("criticmodel_1.h5")
        target_actor.load_weights("actormodel_1.h5")
        target_critic.load_weights("criticmodel_1.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    
    
    ep_hist=[]
    reward_hist = [] # total reward per epoch
    actor_loss_hist = []
    critic_loss_hist = []
    steps_hist = []
    
    
    for i in range(total_episodes):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
            
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
        total_reward = 0.
        total_loss  = 0.
        actor_loss = 0.
        critic_loss = 0.
        steps_per_episode = 0
        
        #print(str(ob.angle)+str(ob.track))
        
        for j in range(total_steps):
            
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            #print(s_t.shape)
            #s_t = s_t.reshape(1, s_t.shape[0])
            #print(s_t.shape)
            #print(type(s_t))
            s_t = tf.expand_dims(tf.convert_to_tensor(s_t, dtype=tf.float32),0)
            #print("egr ",s_t.shape)
            
            # predict action with current state
            a_t_original = actor_model(s_t)
            #print("--------------")
            #print(type(a_t_original) )
            #print("act ", a_t_original.shape)
            
            # add noise for exploration
            noise_t[0][0] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][2], -0.1 , 1.00, 0.05)
            #The following code do the stochastic brake
            if random.random() <= 0.1:
                #print("********Now we apply the brake***********")
                noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.generate_noise(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            
            
            # next step with predicted action, next observation
            ob, r_t, done, info = env.step(a_t[0])
            
            
            
            
            # get next state
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            
            
            s_t1 = tf.convert_to_tensor(s_t1,dtype=tf.float32)
            
            # record current_state, action, reward, next_state, finished?
            buff.add(s_t[0], a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            # draw random sample from buffer of batch size
            batch = buff.getBatch(BATCH_SIZE) 
            
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
            
            #print(states.shape, type(states))
            #print(actions.shape, type(actions))
            #print(new_states.shape, type(new_states))
            target_q_values = target_values(new_states,target_actor,target_critic)
            #print(target_q_values[0])
            #print("eag")
            #print(tf.executing_eagerly())
            
            # discounted Qvalues
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
            #print(y_t[0].dtype)
            
            if (train_indicator):
                
                loss1, loss2 = update(actor_model,critic_model,states,actions,y_t,actor_optimizer,critic_optimizer)
  
                proto_tensor1 = tf.make_tensor_proto(loss1)  # convert `tensor a` to a proto tensor
                loss1 =  tf.make_ndarray(proto_tensor1)
                proto_tensor2 = tf.make_tensor_proto(loss2)  # convert `tensor a` to a proto tensor
                loss2 =  tf.make_ndarray(proto_tensor2)
                
                actor_loss += loss1
                critic_loss += loss2
                
                # update target actor and target critic
                #actor.target_train()
                #critic.target_train()
                update_target(target_actor.variables, actor_model.variables, TAU)
                update_target(target_critic.variables, critic_model.variables, TAU)
                
            total_reward += r_t
            s_t = s_t1 # current state <--- next state
            steps_per_episode = j
            print("Episode", i, "Step", j, "Reward: %.3f"%r_t, "Actor Loss: %.3f"%loss1,"Critic Loss: %.3f"%loss2)
            #if step%100==0:
                #print("Episode", i, "Step", step, "Reward: %.3f"%r_t, "Loss: %.3f"%loss)   
            
            step += 1
            if done:
                break
            
            
        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor_model.save_weights("actormodel_1.h5", overwrite=True)
                with open("actormodel_1.json", "w") as outfile:
                    json.dump(actor_model.to_json(), outfile)

                critic_model.save_weights("criticmodel_1.h5", overwrite=True)
                with open("criticmodel_1.json", "w") as outfile:
                    json.dump(critic_model.to_json(), outfile)
                #print("----------------------\nsaving weights\n------------------------")
        
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        ep_hist.append(i)
        reward_hist.append(total_reward) 
        actor_loss_hist.append(actor_loss)
        critic_loss_hist.append(critic_loss)
        steps_hist.append(steps_per_episode)
        
        
    env.end()  # This is for shutting down TORCS
    print("Finish.")
    
    # DRAWDING EPISODE VS REWARDS VS LOSS
    
    plt.plot(ep_hist,reward_hist,label='reward')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Episode vs Reward without Expert's Knowledge")
    plt.legend()
    plt.savefig('Episode_Vs_Reward_without_Expert_knowledge.png')
    plt.show()
    
    plt.plot(ep_hist,actor_loss_hist,label='actor loss')
    plt.plot(ep_hist,critic_loss_hist,label='critic loss')
    plt.xlabel("Episodes")
    plt.ylabel("Losses")
    plt.title("Episode vs Losses without Expert's Knowledge")
    plt.legend()
    plt.savefig('Episode_Vs_Losses_without_Expert_knowledge.png')
    plt.show()
    
    
    plt.plot(ep_hist,steps_hist,label='steps per episode')
    
    # yet to show: distace covered, time taken
    
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Episode vs Steps without Expert's Knowledge")
    plt.legend()
    plt.savefig('Episode_Vs_Steps_without_Expert_knowledge.png')
    plt.show()
    
if __name__ == "__main__":
    aiAgent(1)
