import numpy as np
import _pickle as pickle
import gym
import cv2

H = 200 # num of hidden layer neuros
batch_size = 10 # update param every batch_size episodes
learning_rate = 1e-3
gamma = 0.99 #discount factor for events after bad/good decision
decay_rate = 0.99 # dont know what this is
resume = False
render = False

D = 80 * 65

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    
grad_buffer = {k:np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k:np.zeros_like(v) for k,v in model.items()}

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def prepro(I):
    I = I[35:195,15:-15] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    #cv2.imshow('pong-vision',I.astype(np.float))
    #cv2.waitKey(1)
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """
    TODO:
    Take into account that negative reward in case of pong is not immediate
    so reducing the reward exponentially after the game is done could be 
    because a bad decision made 2,3 frames in the past.
    - idea: make the gym  stop early
    - idea: factor in some shifted distribution of the discount reward.
    """
    
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        # reset the sum, since this was a game boundary (pong specific!)
        # Pong has either +1 or -1 reward exactly when game ends.
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h

def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    # TODO: Why outer product?
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2': dW2}

env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()
    
    # Preprocessing image and saving difference between frames to capture motion
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    
    #TODO -> discard first observation to obtain actual difference and not some flash
    
    # Forward through policy
    aprob, h = policy_forward(x)
    # sample from distribution given by the probability
    action = 2 if np.random.uniform() < aprob else 3
    
    """
        TODO -> how to sample from softmax?
        - idea: sample each probability for each class
    """
    
    xs.append(x) #save original input
    hs.append(h) #save hidden outputs
    """
    assume 1 is the correct label in a supervised way and remember this was
    retrieved by the sampled distribution.
    The gradient calculated for "dz" which is the derivative of logistic regression loss
    will be multiplied later by a reward discount which encourages gradients that lead to
    successes (discourages gradients that lead to failures)
    """
    y = 1 if action == 2 else 0
    dlogps.append(y-aprob)
    
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    drs.append(reward)
    
    if(done):
        episode_number += 1
        epx = np.vstack(xs) # [num_iterations_until_done, x = 80*80]
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps) # [num_iterations_until_done, 1]
        epr = np.vstack(drs) # [num_iterations_until_done, 1]
        xs, hs, dlogps, drs = [], [], [], [] # Reset episode memory
        
        # Discount rewards
        discounted_epr = discount_rewards(epr)
        
        """
        SOME NORMALIZATION THAT I DON'T UNDERSTAND
        TODO: UNDERSTAND
        """
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        
        # reduce gradient accordingly considering the reward (discounted)
        epdlogp *= discounted_epr
        #TODO: UNDERSTAND THIS BACKPROP
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]
        
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                #rms prop update: remember the weighted moving average across gradients that make them less giggly and converge faster since it moves more in the direction of less change (momentum)
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) #reset gradients for the batch
                
        # Moving average over the reward scores
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'resetting env. episode reward total was ${reward_sum}. running mean: ${running_reward}')
    
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print(f'ep ${episode_number}: game finished, reward: ${reward}')
