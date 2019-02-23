
# Collaboration and Competition

---

In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

### 1. Start the Environment

We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
from unityagents import UnityEnvironment
import numpy as np
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Tennis.app"`
- **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
- **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
- **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
- **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
- **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
- **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`

For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Tennis.app")
```


```python
env = UnityEnvironment(file_name="Tennis.exe")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: TennisBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 3
            Vector Action space type: continuous
            Vector Action space size (per agent): 2
            Vector Action descriptions: , 
    

Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
```

    Number of agents: 2
    Size of each action: 2
    There are 2 agents. Each observes a state with length: 24
    The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.         -6.65278625 -1.5
     -0.          0.          6.83172083  6.         -0.          0.        ]
    

### 3. Take Random Actions in the Environment

In the next code cell, you will learn how to use the Python API to control the agents and receive feedback from the environment.

Once this cell is executed, you will watch the agents' performance, if they select actions at random with each time step.  A window should pop up that allows you to observe the agents.

Of course, as part of the project, you'll have to change the code so that the agents are able to use their experiences to gradually choose better actions when interacting with the environment!


```python
for i in range(1, 600):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        print(actions)
        actions[1][0] = -1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
```

    [[ 0.87568583 -1.        ]
     [ 1.          0.58791733]]
    [[ 1.          0.39029727]
     [-0.86289372  1.        ]]
    [[ 1.          0.39557509]
     [ 1.         -0.12041991]]
    [[-1.         -0.67521886]
     [-1.         -0.71045549]]
    [[-0.3089727   0.29600091]
     [-1.          1.        ]]
    [[ 0.71786931  0.22394413]
     [-0.89437801  0.4707606 ]]
    [[-0.21557358  0.56906155]
     [-1.          1.        ]]
    [[-0.91209722 -0.39889252]
     [-0.29149199 -0.24227932]]
    [[ 1.         -0.45188152]
     [ 0.1261572  -0.30220508]]
    [[-0.85373933  1.        ]
     [-0.77972631  1.        ]]
    [[-0.3376738  -1.        ]
     [-0.73862967  0.20045936]]
    [[-0.67136444  1.        ]
     [ 1.          1.        ]]
    [[-1.          1.        ]
     [ 0.48109235 -1.        ]]
    [[-0.74343526 -0.20310564]
     [ 0.87610092  0.90563038]]
    [[-0.43855349 -0.45687358]
     [-0.38409329  0.45652428]]
    Score (max over agents) from episode 1: 0.0
    [[ 0.7727669  1.       ]
     [-1.         1.       ]]
    [[ 1.          0.51117719]
     [ 0.67133232 -0.22896032]]
    [[ 0.93473246  0.46714517]
     [ 0.50733016 -0.19113876]]
    [[-1.          0.90022608]
     [-1.         -0.11029647]]
    [[ 1.          0.41121748]
     [ 0.91570788 -0.36439019]]
    [[ 0.96624445  1.        ]
     [-1.          1.        ]]
    [[-0.28974741 -1.        ]
     [ 0.23524223 -0.20473607]]
    [[-0.43979184 -1.        ]
     [ 1.         -0.54836344]]
    [[-0.60986282 -1.        ]
     [-0.54305188 -0.91180473]]
    [[-0.55026086 -1.        ]
     [-0.89673276 -0.25750556]]
    [[-0.10084518  0.32249861]
     [ 1.          0.43456921]]
    [[-0.51173127  1.        ]
     [-0.94513363 -0.60433693]]
    [[0.28480167 0.24756482]
     [1.         0.67134267]]
    [[-0.09623647 -0.26331371]
     [ 1.         -0.77787748]]
    Score (max over agents) from episode 2: 0.0
    [[ 1.         -0.52857793]
     [ 1.          0.11142882]]
    [[ 0.34820208 -0.18292997]
     [ 0.54268578 -1.        ]]
    [[-0.35192298  0.31888624]
     [-0.65718949 -0.48777834]]
    [[ 0.56826409  0.18691369]
     [ 0.91919137 -0.15135518]]
    [[-1.         -0.85171313]
     [-1.          0.08911307]]
    [[ 0.25040274  1.        ]
     [-0.59659828 -0.31717867]]
    [[ 1.         -0.78633134]
     [ 0.20293417  1.        ]]
    [[1.         0.72398971]
     [1.         0.97842742]]
    [[ 0.6800036   0.55359942]
     [ 0.5575989  -0.55674771]]
    [[-0.372078   -1.        ]
     [ 0.23905012 -0.04497278]]
    [[-0.93107268 -0.31328356]
     [ 0.14521695 -0.03507144]]
    [[-0.52986344 -1.        ]
     [ 1.          0.97607769]]
    [[ 1.         -0.99202725]
     [ 0.45705308 -0.841109  ]]
    [[ 1.          0.33303932]
     [ 1.         -0.74205622]]
    Score (max over agents) from episode 3: 0.0
    [[-1.          0.76857681]
     [-0.89604966  1.        ]]
    [[-0.7192212   0.6481976 ]
     [-0.21090188  0.59311287]]
    [[ 0.27005882  1.        ]
     [-0.48407028  1.        ]]
    [[-0.97882299  0.39132885]
     [ 0.95041898 -0.60380876]]
    [[ 1.          0.38519924]
     [-0.35657527  0.8705233 ]]
    [[-1.          0.90477912]
     [ 0.51328065 -1.        ]]
    [[-0.25785178 -0.10740455]
     [-0.47883826 -1.        ]]
    [[-0.43522206 -1.        ]
     [-0.42356182  0.8196024 ]]
    [[-1.         -1.        ]
     [-0.5025213   0.64882483]]
    [[ 1.         -0.41717459]
     [ 0.95567216  1.        ]]
    [[ 1.          0.58183179]
     [ 0.51770032 -1.        ]]
    [[ 0.40719191  1.        ]
     [-0.07822016 -0.68520954]]
    [[ 1. -1.]
     [-1. -1.]]
    [[-0.16971116  0.83446453]
     [ 0.46432803 -1.        ]]
    Score (max over agents) from episode 4: 0.0
    [[ 0.19833333 -1.        ]
     [ 0.34725493 -1.        ]]
    [[-1.          0.02045706]
     [-0.28915357  0.25610581]]
    [[-1.          0.54342196]
     [ 1.         -0.16244033]]
    [[ 1.         -0.56185804]
     [-0.45568507  0.24079161]]
    [[-0.08647002  0.86089069]
     [-0.24539697 -0.00903697]]
    [[ 0.28733053 -0.21741004]
     [-0.18805114  1.        ]]
    [[-0.51135238 -1.        ]
     [ 1.         -0.68421435]]
    [[-0.56716994  1.        ]
     [ 1.         -1.        ]]
    [[0.08208298 1.        ]
     [0.62916386 0.51007472]]
    [[ 0.34401712 -1.        ]
     [ 0.9278198   1.        ]]
    [[ 0.61441675  0.11851724]
     [-0.20538163 -0.54959283]]
    [[ 0.60671043 -1.        ]
     [ 0.11838807 -0.0492376 ]]
    [[-1.          0.74650717]
     [ 0.04231895 -0.67780729]]
    [[-0.57829311 -0.61433971]
     [ 0.49614086  0.45413356]]
    Score (max over agents) from episode 5: 0.0
    [[ 0.60142847  1.        ]
     [ 1.         -1.        ]]
    [[-0.37701623  0.93701998]
     [-1.          0.34189602]]
    [[ 0.02204819  0.23455364]
     [-0.94469796  1.        ]]
    [[ 1.          0.72701862]
     [-0.24340734  0.71307524]]
    [[-0.32857967  0.80322582]
     [ 0.25223937  1.        ]]
    [[ 0.29315969  1.        ]
     [-1.          1.        ]]
    [[-0.54374387 -0.84210699]
     [ 0.33499587  1.        ]]
    [[ 1.         -0.64036101]
     [ 1.         -0.76450797]]
    [[ 0.1310434  -0.64828938]
     [ 1.         -0.40415416]]
    [[-0.53830461 -1.        ]
     [ 0.0270528  -0.39355291]]
    [[-0.57140664 -0.2994787 ]
     [-1.          0.40011672]]
    [[-0.6121993   1.        ]
     [ 0.29023953 -1.        ]]
    [[ 0.75382667 -1.        ]
     [-0.30815314 -1.        ]]
    [[-0.2673921  -0.94120702]
     [ 0.043451    0.88656361]]
    [[ 0.30564599  0.04327116]
     [-1.         -0.09404405]]
    [[ 0.44096441 -0.54503504]
     [-0.65444048  0.61267953]]
    Score (max over agents) from episode 6: 0.0
    [[ 1.         -1.        ]
     [-0.17786668 -0.66791565]]
    [[-0.82190191 -1.        ]
     [-0.48037628  1.        ]]
    [[ 0.41735255 -0.296203  ]
     [-0.23273833 -0.87631682]]
    [[ 0.29187405 -0.33203203]
     [ 1.         -0.39754744]]
    [[-1.         -0.10661026]
     [ 0.18468601 -0.66439715]]
    [[ 1.          0.44971953]
     [-0.56542471  1.        ]]
    [[ 0.50849772 -1.        ]
     [-0.17906993  0.27710256]]
    [[-0.69926793 -0.15224485]
     [ 0.15475986  0.86179469]]
    [[ 0.47822116  0.30194216]
     [-0.38724924 -0.14621311]]
    [[ 0.85776956 -0.13082819]
     [ 1.          0.51793925]]
    [[-0.36695985 -0.51114559]
     [ 0.41325692  0.15852916]]
    [[-0.78692634 -0.49629729]
     [ 1.         -0.96943094]]
    [[ 1.          0.13787935]
     [ 0.27998395 -1.        ]]
    [[-0.48149048  0.63094322]
     [-0.05819262  0.72571473]]
    [[ 0.85472088  0.41805184]
     [ 0.28764665 -0.51891222]]
    Score (max over agents) from episode 7: 0.0
    [[-1.          0.44044962]
     [-0.18497129  1.        ]]
    [[-0.12365783  1.        ]
     [-1.         -0.97090318]]
    [[ 0.11817046 -0.13042732]
     [-0.04653737 -0.44455491]]
    [[-0.09065743  1.        ]
     [ 1.         -0.01843013]]
    [[ 0.0627839   1.        ]
     [-0.12014094 -0.01513194]]
    [[-0.16809649  0.43166184]
     [ 1.          1.        ]]
    [[-0.58200777 -0.84504722]
     [ 0.11270425 -0.13006236]]
    [[ 0.86696964 -0.29288208]
     [ 0.84007229  1.        ]]
    [[ 0.90914215 -0.1030625 ]
     [-1.         -0.35505508]]
    [[-1.         -0.57612378]
     [-1.         -0.48747433]]
    [[-0.04187025  0.65307326]
     [ 0.49817165 -1.        ]]
    [[-0.07009993  0.10139189]
     [-1.         -0.19477053]]
    [[ 0.39098097 -0.06883818]
     [-0.16843199 -0.1257607 ]]
    [[1.         0.50797508]
     [0.2304652  1.        ]]
    [[ 0.00850599 -0.57460287]
     [-0.55546313  0.27911452]]
    [[ 1.          0.28285872]
     [-0.68637384  0.74256169]]
    [[-1.          0.19482709]
     [ 1.         -0.08059601]]
    [[-0.37580468  0.48089505]
     [-0.04722775  1.        ]]
    [[1.         0.44844445]
     [0.54152419 0.7818516 ]]
    [[ 1.         -0.40469665]
     [-1.         -1.        ]]
    [[-0.61403918  1.        ]
     [ 0.69202587 -1.        ]]
    [[-1. -1.]
     [-1.  1.]]
    [[-0.13099524  0.34932712]
     [-0.0512883  -0.67663857]]
    [[ 1.         -0.35661171]
     [ 0.44585251 -0.85900165]]
    [[-0.2893266   1.        ]
     [ 0.89328881  0.07256642]]
    [[ 0.92320419 -0.2536317 ]
     [ 1.         -0.62039022]]
    [[-0.02374054  0.95537396]
     [ 0.44982087  0.85106694]]
    [[-0.76764863 -1.        ]
     [ 0.74211822 -0.55673581]]
    [[-0.46956714 -0.13690372]
     [-0.19188464 -0.21989571]]
    Score (max over agents) from episode 8: 0.10000000149011612
    [[-0.01292176  0.65728824]
     [ 0.44203202  0.98587431]]
    [[-0.85205043  1.        ]
     [ 0.95818319  0.55315428]]
    [[-0.83190853 -1.        ]
     [ 1.         -1.        ]]
    [[ 0.91899443 -0.30640015]
     [-1.         -0.2328512 ]]
    [[ 0.65288298 -1.        ]
     [ 0.64417757  1.        ]]
    [[ 0.34027001 -1.        ]
     [-0.93185715 -0.53912939]]
    [[ 0.96010412 -0.21734865]
     [-0.10231635 -0.02220553]]
    [[ 0.4287354   0.69039781]
     [-1.          0.70417311]]
    [[ 0.88981843 -0.89691815]
     [-0.00326346 -0.71384748]]
    [[ 3.98473342e-04  1.00000000e+00]
     [ 1.67485644e-01 -1.00000000e+00]]
    [[-1.          0.86062345]
     [-0.89493557  0.03820811]]
    [[-1.         -0.47762066]
     [-0.60085862 -1.        ]]
    [[1.         1.        ]
     [0.40752346 1.        ]]
    [[ 0.20236868  1.        ]
     [ 1.         -0.28112366]]
    Score (max over agents) from episode 9: 0.0
    [[ 0.6842383 -1.       ]
     [-0.1187617  1.       ]]
    [[ 1.         -0.61569352]
     [ 0.20755428  0.26099504]]
    [[-1.          0.40443455]
     [-0.8140033  -0.91631507]]
    [[-0.0074944  -0.68675176]
     [-0.37342789  0.46451673]]
    [[-0.9917933   1.        ]
     [ 1.          0.89252532]]
    [[-0.63566519  0.02781325]
     [ 1.         -0.81143271]]
    [[-0.46555935 -0.18634051]
     [-1.         -1.        ]]
    [[ 1.         -0.41647182]
     [-1.         -0.10946215]]
    [[ 0.20654436 -0.0399723 ]
     [-1.          0.67577889]]
    [[ 1.          1.        ]
     [-1.          0.20767915]]
    [[ 1.         -0.50763792]
     [-0.64782194 -0.12582193]]
    [[ 0.0888763  -1.        ]
     [ 0.24304384 -0.59897028]]
    [[ 0.2167623   1.        ]
     [ 1.         -0.44316943]]
    [[ 0.70590429 -1.        ]
     [-1.          0.65172731]]
    Score (max over agents) from episode 10: 0.0
    [[ 0.10637348 -1.        ]
     [ 1.          0.63529601]]
    [[-0.09747855  0.67516464]
     [ 0.66982016  0.22865279]]
    [[ 0.48737691 -1.        ]
     [ 0.41910743  0.15827064]]
    

When finished, you can close the environment.


```python
env.close()
```

### 4. It's Your Turn!

Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
```python
env_info = env.reset(train_mode=True)[brain_name]
```


```python
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline
from maddpg_agent import Agent
agents = Agent(state_size=state_size, action_size=action_size, random_seed=2)
def ddpg(n_episodes=11000, max_t=10000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        score = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agents.reset()
        x=0
        for t in range(max_t):
            action = agents.act(states)
            env_info = env.step(action)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            score += rewards
            dones = env_info.local_done
            agents.step(states, action, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores.append(score)
        if i_episode%print_every==0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agents.actor_local1.state_dict(), 'checkpoint_actor1.pth')
        torch.save(agents.critic_local1.state_dict(), 'checkpoint_critic1.pth')
        torch.save(agents.actor_local2.state_dict(), 'checkpoint_actor2.pth')
        torch.save(agents.critic_local2.state_dict(), 'checkpoint_critic2.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

    Episode 100	Average Score: 0.01
    Episode 200	Average Score: 0.01
    Episode 300	Average Score: 0.01
    Episode 400	Average Score: 0.01
    Episode 500	Average Score: 0.00
    Episode 600	Average Score: 0.00
    Episode 700	Average Score: 0.01
    Episode 800	Average Score: 0.01
    Episode 900	Average Score: 0.01
    Episode 1000	Average Score: 0.00
    Episode 1100	Average Score: 0.01
    Episode 1200	Average Score: 0.02
    Episode 1300	Average Score: 0.02
    Episode 1400	Average Score: 0.02
    Episode 1500	Average Score: 0.03
    Episode 1600	Average Score: 0.03
    Episode 1700	Average Score: 0.04
    Episode 1800	Average Score: 0.04
    Episode 1900	Average Score: 0.05
    Episode 2000	Average Score: 0.05
    Episode 2100	Average Score: 0.05
    Episode 2200	Average Score: 0.06
    Episode 2300	Average Score: 0.05
    Episode 2400	Average Score: 0.05
    Episode 2500	Average Score: 0.06
    Episode 2600	Average Score: 0.06
    Episode 2700	Average Score: 0.06
    Episode 2800	Average Score: 0.04
    Episode 2900	Average Score: 0.05
    Episode 3000	Average Score: 0.05
    Episode 3100	Average Score: 0.05
    Episode 3200	Average Score: 0.04
    Episode 3300	Average Score: 0.06
    Episode 3400	Average Score: 0.06
    Episode 3500	Average Score: 0.06
    Episode 3600	Average Score: 0.05
    Episode 3700	Average Score: 0.06
    Episode 3800	Average Score: 0.09
    Episode 3900	Average Score: 0.07
    Episode 4000	Average Score: 0.07
    Episode 4100	Average Score: 0.08
    Episode 4200	Average Score: 0.08
    Episode 4300	Average Score: 0.09
    Episode 4400	Average Score: 0.09
    Episode 4500	Average Score: 0.10
    Episode 4600	Average Score: 0.09
    Episode 4700	Average Score: 0.11
    Episode 4800	Average Score: 0.11
    Episode 4900	Average Score: 0.11
    Episode 5000	Average Score: 0.13
    Episode 5100	Average Score: 0.12
    Episode 5200	Average Score: 0.16
    Episode 5300	Average Score: 0.18
    Episode 5400	Average Score: 0.22
    Episode 5500	Average Score: 0.30
    Episode 5600	Average Score: 0.39
    Episode 5700	Average Score: 0.44
    Episode 5800	Average Score: 0.36
    Episode 5900	Average Score: 0.38
    Episode 6000	Average Score: 0.36
    Episode 6100	Average Score: 0.32
    Episode 6200	Average Score: 0.42
    Episode 6300	Average Score: 0.32
    Episode 6400	Average Score: 0.45
    Episode 6500	Average Score: 0.51
    Episode 6600	Average Score: 0.54
    Episode 6700	Average Score: 0.50
    Episode 6800	Average Score: 0.47
    Episode 6900	Average Score: 0.52
    Episode 7000	Average Score: 0.42
    Episode 7100	Average Score: 0.43
    Episode 7200	Average Score: 0.49
    Episode 7300	Average Score: 0.60
    Episode 7400	Average Score: 0.67
    Episode 7500	Average Score: 0.78
    Episode 7600	Average Score: 0.69
    Episode 7700	Average Score: 0.72
    Episode 7800	Average Score: 0.59
    Episode 7900	Average Score: 0.67
    Episode 8000	Average Score: 0.61
    Episode 8100	Average Score: 0.60
    Episode 8200	Average Score: 0.77
    Episode 8300	Average Score: 0.70
    Episode 8400	Average Score: 0.58
    Episode 8500	Average Score: 0.79
    Episode 8600	Average Score: 0.79
    Episode 8700	Average Score: 0.66
    Episode 8800	Average Score: 0.48
    Episode 8900	Average Score: 0.41
    Episode 9000	Average Score: 0.49
    Episode 9100	Average Score: 0.52
    Episode 9200	Average Score: 0.59
    Episode 9300	Average Score: 0.63
    Episode 9400	Average Score: 0.59
    Episode 9500	Average Score: 0.63
    Episode 9600	Average Score: 0.41
    Episode 9700	Average Score: 0.42
    Episode 9800	Average Score: 0.68
    Episode 9900	Average Score: 0.58
    Episode 10000	Average Score: 0.58
    Episode 10100	Average Score: 0.66
    Episode 10200	Average Score: 0.62
    Episode 10300	Average Score: 0.52
    Episode 10400	Average Score: 0.36
    Episode 10500	Average Score: 0.40
    Episode 10600	Average Score: 0.76
    Episode 10700	Average Score: 1.01
    Episode 10800	Average Score: 0.80
    Episode 10900	Average Score: 0.62
    Episode 11000	Average Score: 0.70
    


![png](output_13_1.png)



```python
import torch
from maddpg_agent import Agent
agents = Agent(state_size, action_size, random_seed=2)
agents.actor_local1.load_state_dict(torch.load('checkpoint_actor1.pth'))
agents.critic_local1.load_state_dict(torch.load('checkpoint_critic1.pth'))
agents.actor_local2.load_state_dict(torch.load('checkpoint_actor2.pth'))
agents.critic_local2.load_state_dict(torch.load('checkpoint_critic2.pth'))
for i in range(1, 20):
    scores = np.zeros(num_agents)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    while True:
        action = agents.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        scores += env_info.rewards
        state = next_state
        if np.any(done):
            break 

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```

    Total score (averaged over agents) this episode: 0.4950000075623393
    Total score (averaged over agents) this episode: 1.145000017248094
    Total score (averaged over agents) this episode: -0.004999999888241291
    Total score (averaged over agents) this episode: 0.7450000112876296
    Total score (averaged over agents) this episode: 0.1450000023469329
    Total score (averaged over agents) this episode: 0.09500000160187483
    Total score (averaged over agents) this episode: 0.245000003837049
    Total score (averaged over agents) this episode: 0.9950000150129199
    Total score (averaged over agents) this episode: 0.7450000112876296
    Total score (averaged over agents) this episode: 0.29500000458210707
    Total score (averaged over agents) this episode: 0.44500000681728125
    Total score (averaged over agents) this episode: 0.6450000097975135
    Total score (averaged over agents) this episode: 0.09500000160187483
    Total score (averaged over agents) this episode: 2.1950000328943133
    Total score (averaged over agents) this episode: 0.09500000160187483
    Total score (averaged over agents) this episode: 0.8950000135228038
    Total score (averaged over agents) this episode: 2.600000038743019
    Total score (averaged over agents) this episode: 2.5450000381097198
    Total score (averaged over agents) this episode: 0.04500000085681677
    


```python

```


```python

```


```python

```
