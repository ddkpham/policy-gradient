import tensorflow as tf 
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym
import matplotlib.pyplot as plt


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
    """
    advantage = []
    
    return advantage

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
        print('reshaped state', state)
        print('action_probs', self.action_probs)
        probs = self.sess.run([self.action_probs], feed_dict={self.state: state})[0]
        print('probs: ', probs)
        return probs

    def update(self, gradient_buffer):
        feed  = dict(zip(self.gradient_holder, gradient_buffer))
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

    # def get_action(self, state):
    #     #convert state to correct float precision 
    #     converted_state = np.array(state, dtype=np.float32)
    #     #test = tf.constant([[state[0]], [state[1]], [state[2]], [state[3]]])
    #     test = tf.constant(state)
    #     print('test',converted_state)
    #     action_probabilities = model(test)
    #     action = tf.math.argmax(action_probabilities)
    #     return action
# --- end class ---

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()


##### Main script #####
env = gym.make('CartPole-v1') # Initialize gym environment
gamma = 0.99                  # Discount factor

# initialize tensor flow model
tf.reset_default_graph()

### -----------------------------------

total_episodes = 2500 # maximum # of training episodes
max_steps = 500 # maximum # of steps per episode (overridden in gym environment)
update_frequency = 5 # number of episodes between policy network updates

# Begin tensorflow session
# init = tf.global_variables_initializer()   
with tf.Session() as sess:
    init = tf.global_variables_initializer()  
    # Initialization
    # myAgent = agent(sess, env, 0.001, 4, 2, 8, 8 ) 
    myAgent = policy_estimator(sess,env)
    sess.run(init)
    i = 0
    ### --- Part g) create the RL agent ---
    # uncomment fill in arguments of the above line to initialize an RL agent whose
    # policy network contains two hidden layers with 8 neurons each
    
    ep_rewards = []
    history = []
    
    while i < total_episodes:
        # reset environment
        s = env.reset() 
        # Get possible actions
        action_space = np.arange(env.action_space.n)

        for j in range(max_steps):
            # Visualize behaviour every 100 episodes
            if i % 100 == 0:
                env.render()
            # --- end if ---
            print('action_space', env.action_space)
            print('environment_space', env.observation_space, env.observation_space)
            
            ### ------------ Part g) -------------------
            ### Probabilistically pick an action given policy network outputs.
            
            ### ----------------------------------------
            print('state: ', s)
            action_probs = myAgent.predict(s.reshape(1,-1))
            # action = np.random.choice(action_space, p=action_probs)
            action = 1
            print('action: ', action)
            # action = env.action_space.sample()
            # Get reward for taking an action, and store data from the episode
            s1,reward,done,info = env.step(action) #
            history.append([j,s,action,reward,s1])
            s = s1

            if done == True: # Update the network when episode is done
                # Update network every "update_frequency" episodes
                if i % update_frequency == 0 and i != 0:
                    # Compute advantage
                    history = np.array(history)
                    advantage = compute_advantage(history[:,0], history[:,3], \
                                                  gamma)
                    
                    ### --- Part g) Perform policy update ---
                    
                    ### -------------------------------------
                    
                    # Reset history
                    history = []
                # --- end if ---

                break
            # --- end if ---
        # --- end for ---

        i += 1
    # --- end while ---
    
# --- end of script ---