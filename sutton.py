import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import warnings

def ep_arr(j, reward):
  reward_list = []
  ep_reward = []
  episode_list = []
  episode = []
  for index, el in enumerate(j):
    if(index == 0):
      episode.append(el)

      ep_reward.append(reward[index])
    elif(el == 0):
      episode_list.append(episode)
      episode = [0]

      reward_list.append(ep_reward)
      ep_reward = [reward[index]]
    elif(index == len(j)-1):
      episode.append(el)
      episode_list.append(episode)

      ep_reward.append(reward[index])
      reward_list.append(ep_reward)
    else:
      episode.append(el)
      
      ep_reward.append(reward[index])

  return episode_list, reward_list


def compute_advantage(j, reward, gamma):
    ### Part f) Advantage computation
    """ Computes the advantage function from data
        Inputs:
            j     -- list of time steps 
                    (eg. j == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5] means that there 
                     are two episodes, one with four time steps and another with
                     6 time steps)
            reward     -- list of rewards from episodes corresponding to time steps j
            gamma -- discount factor
        
        Output:
            advantage -- vector of advantages correponding to time steps j
            where each time step is equal to 
            the sum of the future discounted rewards from time t onwards
    """

    episode_list, reward_list = ep_arr(j, reward)
    #print('episode_list: ', episode_list, len(episode_list))
    #print('reward_list: ', reward_list, len(reward_list))
    advantage = []

    for index, ep in enumerate(episode_list):
      # calculate value approximation for episode b
      b = 0
      gamma_t = 1
      for t in range(len(ep)):
        b += gamma_t * reward_list[index][t]
        gamma_t *= gamma
      # calculate reward of timestep
      for t in ep:
        j = t
        discounted_reward = 0
        gamma_t = 1
        while(j < len(ep)):
          discounted_reward += gamma_t * reward_list[index][j]
          gamma_t *= gamma
          adv_t = discounted_reward - b
          j += 1
        advantage.append(adv_t)

    # print('advantage: ', advantage, len(advantage))
    return advantage

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

class policy_estimator(object):
    
    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = 0.01
        
        # Define number of hidden nodes
        self.n_hidden_nodes = 16
        
        # Set graph scope name
        self.scope = "policy_estimator"
        
        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()
            
            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs], 
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            
            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            # Get probability of each action
            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))
            
            # Get indices of actions
            indices = tf.range(0, tf.shape(output_layer)[0]) \
                * tf.shape(output_layer)[1] + self.actions
                
            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]),
                                             indices)
    
            # Define loss function
            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, 
                    name='grads' + str(j)))
            
            self.gradients = tf.gradients(self.loss, self.tvars)
            
            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))
            
    def predict(self, state):
        probs = self.sess.run([self.action_probs], 
                              feed_dict={
                                  self.state: state
                              })[0]
        return probs
    
    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.actions: actions,
            self.rewards: rewards
            })[0]
        return grads 

class agent():
    def __init__(self, sess, env, lr, s_size, a_size, h1_size, h2_size):
        """ Initialize the RL agent 
        Inputs:
            lr      -- learning rate
            s_size  -- # of states
            a_size  -- # of actions (output of policy network)
            h1_size -- # of neurons in first hidden layer of policy network
            h2_size -- # of neurons in second hidden layer of policy network
        """
        self.sess = sess
        self.env = env
        self.lr = lr
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.n
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.scope = "my-agent"
        
        print(env.observation_space.shape[0])
        print(env.action_space.n)
        
        # Data consists of a list of states, actions, and rewards
        self.advantage_data = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_data = tf.placeholder(shape=[None], dtype=tf.int32)        
        self.state_data = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        
        ### --- Part c) Define the policy network ---
        # Input should be the state (defined above)
        # Output should be the probability distribution of actions
    
        # Create NN
        with tf.variable_scope(self.scope):
            print('creating model.....')
            # initializes all weights in the network
            initializer = tf.contrib.layers.xavier_initializer()

            # Define place holder tensors
            self.state = tf.placeholder(tf.float32, [None, self.s_size], name="state")
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

            # Create layers 
            # layer_1 = fully_connected(self.state, h1_size, activation_fn=tf.nn.relu, weights_initializer=initializer)
            # layer_2 = fully_connected(layer_1, h2_size, activation_fn=tf.nn.relu, weights_initializer=initializer)
            # output_layer = fully_connected(layer_2, self.a_size, activation_fn=None, weights_initializer=initializer)

            layer_1 = fully_connected(self.state, self.h1_size,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.a_size,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            # Get probability of each action
            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))
            # Get the action probabilities 
            # self.action_probs = tf.squeeze(tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))
            # self.action_probs = tf.nn.softmax(output_layer - tf.reduce_max(output_layer))
            print('shape', tf.shape(output_layer)[0], tf.shape(output_layer)[1])
            # Get indices of actions
            indices = tf.range(0, tf.shape(output_layer)[0]) \
                * tf.shape(output_layer)[1] + self.actions


            print('indices', indices)
            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]), indices)

            # Computes - mean of loss
            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            # Get gradients and variables 
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, name="grads" + str(j)))

            self.gradients = tf.gradients(self.loss, self.tvars)

            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))

        
        ### -----------------------------------------
        
        ### -- Part d) Compute probabilities of realized actions (from data) --
        # Indices of policy network outputs (which are probabilities) 
        # corresponding to action data

        # def get_action(self, x):

        
        ### -------------------------------------------------------------------
        
        ### -- Part e) Define loss function for policy improvement procedure --

        
        ### -------------------------------------------------------------------
        
        # # Gradient computation
        # tvars = tf.trainable_variables()
        # self.gradients = tf.gradients(self.loss, tvars)
        
        # # Apply update step
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.update_batch = optimizer.apply_gradients(zip(self.gradients, tvars))
    # --- end def ---

    def predict(self, state):
        probs = self.sess.run([self.action_probs], feed_dict={self.state: state})[0]
        return probs

    def update(self, gradient_buffer):
        feed  = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    # returns a list of all trainable variables, i.e weights and biases 
    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    # returns all gradients 
    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.actions: actions,
            self.rewards: rewards
            })[0]
        return grads   

# --- end class ---


def reinforce_mo(env, policy_estimator, total_episodes=2000,
              batch_size=10, gamma=0.99):
  # Discount factor
  total_rewards = []
  gamma = 0.99                  
  total_episodes = 10000 # maximum # of training episodes original 2500
  max_steps = 500 # maximum # of steps per episode (overridden in gym environment)
  update_frequency = 5 # number of episodes between policy network updates  

  # Set up gradient buffers and set values to 0
  grad_buffer_pe = policy_estimator.get_vars()
  for i, g in enumerate(grad_buffer_pe):
      grad_buffer_pe[i] = g * 0

  with tf.Session() as sess:
    init = tf.global_variables_initializer()  
    # Get possible actions
    action_space = np.arange(env.action_space.n)
    myAgent = policy_estimator
    sess.run(init)
    i = 0
    ### --- Part g) create the RL agent ---
    # uncomment fill in arguments of the above line to initialize an RL agent whose
    # policy network contains two hidden layers with 8 neurons each
    
    ep_rewards = []
    history = []
    
    while i < total_episodes:
        # reset environment
        if i % 100 == 0:
          print('episode : ', i)
        
        s = env.reset() 
        for j in range(max_steps):
            # Visualize behaviour every 100 episodes
            if i % 100 == 0:
                env.render()
              
            # --- end if ---
            
            ### ------------ Part g) -------------------
            ### Probabilistically pick an action given policy network outputs.
            action_probs = myAgent.predict(s.reshape(1,-1))
            action = np.random.choice(action_space, p=action_probs)
            ### ----------------------------------------
            s1,reward,done,info = env.step(action) #
            history.append([j,s,action,reward,s1])
            s = s1

            if done == True: # Update the network when episode is done
                # Update network every "update_frequency" episodes
                if i % update_frequency == 0 and i != 0:
                    # Compute advantage
                    history = np.array(history)
                    total_rewards.append(history[:,3].sum())
                    advantage = compute_advantage(history[:,0], history[:,3], \
                                                  gamma)
                    # advantage = discount_rewards(
                    # history[:,3], gamma)

                    # Calculate the gradients for the policy estimator and
                    # add to buffer
                    pe_grads = policy_estimator.get_grads(
                        states=np.vstack(history[:,1]),
                        actions=history[:,2],
                        rewards=advantage)
                    for k, g in enumerate(pe_grads):
                        grad_buffer_pe[k] += g
                    
                    ### --- Part g) Perform policy update ---
                    policy_estimator.update(grad_buffer_pe)
                    # Clear buffer values for next batch
                    for k, g in enumerate(grad_buffer_pe):
                        grad_buffer_pe[k] = g * 0
                    ### -------------------------------------
                    
                    # Reset history
                    history = []
                # --- end if ---
                break
            # --- end if ---
        # --- end for ---
        i += 1
    # --- end while ---
    return total_rewards
# --- end of script ---

def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    
    total_rewards = []
    
    # Set up gradient buffers and set values to 0
    grad_buffer_pe = policy_estimator.get_vars()
    for i, g in enumerate(grad_buffer_pe):
        grad_buffer_pe[i] = g * 0
        
    # Get possible actions
    action_space = np.arange(env.action_space.n)
        
    for ep in range(num_episodes):
        # Get initial state
        s_0 = env.reset()
        reward = 0
        episode_log = []
        complete = False
        if(ep % 100 ==0):
          env.render()
        
        # Run through each episode
        while complete == False:
            
            # Get the probabilities over the actions
            action_probs = policy_estimator.predict(
                s_0.reshape(1,-1))
            # Stochastically select the action
            action = np.random.choice(action_space,
                                      p=action_probs)
            # Take a step
            s_1, r, complete, _ = env.step(action)
            
            # Append results to the episode log
            episode_log.append([s_0, action, r, s_1])
            s_0 = s_1
            
            # If complete, store results and calculate the gradients
            if complete:
                episode_log = np.array(episode_log)
                
                # Store raw rewards and discount episode rewards
                total_rewards.append(episode_log[:,2].sum())
                discounted_rewards = discount_rewards(
                    episode_log[:,2], gamma)
                
                # Calculate the gradients for the policy estimator and
                # add to buffer
                pe_grads = policy_estimator.get_grads(
                    states=np.vstack(episode_log[:,0]),
                    actions=episode_log[:,1],
                    rewards=discounted_rewards)
                for i, g in enumerate(pe_grads):
                    grad_buffer_pe[i] += g
                    
        # Update policy gradients based on batch_size parameter
        if ep % batch_size == 0 and ep != 0:
            policy_estimator.update(grad_buffer_pe)
            # Clear buffer values for next batch
            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0
                
    return total_rewards

env = gym.make('CartPole-v1')
tf.reset_default_graph()
sess = tf.Session()

# pe = policy_estimator(sess, env)
pe = agent(sess, env, 0.001, 4, 2, 8, 8 )

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

rewards = reinforce_mo(env, pe)
print(rewards)