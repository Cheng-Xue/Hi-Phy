
# Hi-Phy: A Benchmark for Hierarchical Physical Reasoning
<p align="center">
Cheng Xue*, Vimukthini Pinto*, Chathura Gamage*, Peng Zhang, Jochen Renz<br>
School of Computing<br>
The Australian National University<br>
Canberra, Australia<br>
{cheng.xue, vimukthini.inguruwattage, chathura.gamage}@anu.edu.au<br>
{p.zhang, jochen.renz}@anu.edu.au
 </p>

Reasoning about the behaviour of physical objects is a key capability of agents operating in physical worlds. Humans are
very experienced in physical reasoning while it remains a major challenge for AI. To facilitate research addressing this
problem, several benchmarks have been proposed recently. However, these benchmarks do not enable us to measure an
agent's granular physical reasoning capabilities when solving a complex reasoning task. In this paper, we propose a new
benchmark for physical reasoning that allows us to test individual physical reasoning capabilities. Inspired by how
humans acquire these capabilities, we propose a general hierarchy of physical reasoning capabilities with increasing
complexity. Our benchmark tests capabilities according to this hierarchy through generated physical reasoning tasks in
the video game Angry Birds. This benchmark enables us to conduct a comprehensive agent evaluation by measuring the
agent's granular physical reasoning capabilities. We conduct an evaluation with human players, learning agents, and
heuristic agents and determine their capabilities. Our evaluation shows that learning agents, with good local
generalization ability, still struggle to learn the underlying physical reasoning capabilities and perform worse than
current state-of-the-art heuristic agents and humans. We believe that this benchmark will encourage researchers to
develop intelligent agents with advanced, human-like physical reasoning capabilities.

\* equal contribution

---
#### Table of contents
1. [Hierarchy](#Hierarchy)
2. [Hi-Phy in Angry Birds](#Hi-Phy)
3. [Task Generator](#Task-generator)
4. [Tasks generted for the baseline analysis](#Tasks-generated-for-baselines)
5. [Baseline Agents](#BAF)
   1. [How to run heuristic agents](#RHA)
   2. [How to run DQN Baseline](#RLA)
   3. [How to develop your own agent](#ROA)
   4. [Outline of the Agent Code](#code)
6. [Framework](#Framework)
   1. [The Game Environment](#Env)
   2. [Symbolic Representation Data Structure](#SymbolicRepresentation)
   3. [Communication Protocols](#Protocol)
---

---



## 1.Hierarchy 
<a name="Hierarchy"/></a>
Humans and AI approaches learn much better when examples are presented in a meaningful order with increasing complexity
than when examples are presented randomly. We thereby propose a hierarchy for physical reasoning that enables an agent
to start with increasing complexity to facilitate training and evaluating agents to work in the real physical world.

Our hierarchy consists three levels and fifteen capabilities:

**Level 1: Understanding the instant effect of the first force applied to objects in an environment as a result of an
agent's action.**

    Level 1 capabilities:
    1.1: Understanding the instant effect of objects in an enviornment when an agent applys a single force.
    1.2: Unverstanding the instant effect of objects in an enviornment when an agent applys a multiple force.

**Level 2: Understanding objects movement in the environment after a force is applied.**

    Level 2 capabilities:
    2.1: Understanding that objects in the enviornment may roll.
    2.2: Understanding that objects in the enviornment may fall.
    2.3: Understanding that objects in the enviornment may slide.
    2.4: Understanding that objects in the enviornment may bounce.

**Level 3: Performing in tasks that require capabilities 1) human developed in infancy, 2) required in robotics to develop agents that
work alongside people, and 3) currently fall short in reinforcement learning.**

    Level 3 capabilities:
    3.1: Understanding relative weight of objects.
    3.2: Understanding relative height of objects.
    3.3: Understanding relative width of objects.
    3.4: Understanding shape difference of objects.
    3.5: Understanding how to preform non-greedy actions.
    3.6: Understanding structural weak points/stability.
    3.7: Understanding how to clear a path towards the goal.
    3.8: Understanding how to preform action with adequate timing.
    3.9: Understanding how to use tools.

Please refer to the paper for more details how and why we attributed the capabilities in this way.

## 2. Hi-Phy in Angry Birds
<a name="Hi-Phy"/></a>
Based on the proposed hierarchy, we develop Hi-Phy benchmark in Angry Birds. Hi-Phy contains tasks from 65 task templates belonging to the fifteen capabilities. Shown below are fifteen example tasks in Hi-Phy representing the fifteen capabilities and the solutions for those tasks.

| Task             |  Description |
:-------------------------:|:-----------
<img src="tasks/example_tasks/videos/1.1.1.gif" width="500"/>  |  1.1: Understanding the instant effect of objects in an enviornment when an agent applys a single force. Agent needs to understand that applying a force to the pig (green-coloured object) destroys the pig and solves the task.

<img src="tasks/example_tasks/videos/1.1.1.gif" width="250"/> <img src="tasks/example_tasks/videos/1.2.2.gif" width="250"/> <img src="tasks/example_tasks/videos/2.1.4.gif" width="250"/> 
<img src="tasks/example_tasks/videos/2.2.1.gif" width="250"/> <img src="tasks/example_tasks/videos/2.3.1.gif" width="250"/> <img src="tasks/example_tasks/videos/2.4.2.gif" width="250"/> 
<img src="tasks/example_tasks/videos/3.1.3.gif" width="250"/> <img src="tasks/example_tasks/videos/3.2.3.gif" width="250"/> <img src="tasks/example_tasks/videos/3.3.3.gif" width="250"/> 
<img src="tasks/example_tasks/videos/3.4.3.gif" width="250"/> <img src="tasks/example_tasks/videos/3.5.5.gif" width="250"/> <img src="tasks/example_tasks/videos/3.6.5.gif" width="250"/> 
<img src="tasks/example_tasks/videos/3.7.5.gif" width="250"/> <img src="tasks/example_tasks/videos/3.8.1.gif" width="250"/> <img src="tasks/example_tasks/videos/3.9.4.gif" width="250"/> 

Sceenshots of the 65 task templates are shown below.

<img src="tasks/example_tasks/images/1.1.1.png" width="250"/> <img src="tasks/example_tasks/images/1.1.2.png" width="250"/> <img src="tasks/example_tasks/images/1.1.3.png" width="250"/> 
<img src="tasks/example_tasks/images/1.2.1.png" width="250"/> <img src="tasks/example_tasks/images/1.2.2.png" width="250"/> <img src="tasks/example_tasks/images/1.2.3.png" width="250"/> 
<img src="tasks/example_tasks/images/1.2.4.png" width="250"/> <img src="tasks/example_tasks/images/1.2.5.png" width="250"/> <img src="tasks/example_tasks/images/2.1.1.png" width="250"/> 
<img src="tasks/example_tasks/images/2.1.2.png" width="250"/> <img src="tasks/example_tasks/images/2.1.3.png" width="250"/> <img src="tasks/example_tasks/images/2.1.4.png" width="250"/> 
<img src="tasks/example_tasks/images/2.1.5.png" width="250"/> <img src="tasks/example_tasks/images/2.2.1.png" width="250"/> <img src="tasks/example_tasks/images/2.2.2.png" width="250"/> 
<img src="tasks/example_tasks/images/2.2.3.png" width="250"/> <img src="tasks/example_tasks/images/2.2.4.png" width="250"/> <img src="tasks/example_tasks/images/2.2.5.png" width="250"/> 
<img src="tasks/example_tasks/images/2.3.1.png" width="250"/> <img src="tasks/example_tasks/images/2.3.2.png" width="250"/> <img src="tasks/example_tasks/images/2.3.3.png" width="250"/> 
<img src="tasks/example_tasks/images/2.3.4.png" width="250"/> <img src="tasks/example_tasks/images/2.4.1.png" width="250"/> <img src="tasks/example_tasks/images/2.4.2.png" width="250"/> 
<img src="tasks/example_tasks/images/2.4.3.png" width="250"/> <img src="tasks/example_tasks/images/3.1.1.png" width="250"/> <img src="tasks/example_tasks/images/3.1.2.png" width="250"/> 
<img src="tasks/example_tasks/images/3.1.3.png" width="250"/> <img src="tasks/example_tasks/images/3.1.4.png" width="250"/> <img src="tasks/example_tasks/images/3.1.5.png" width="250"/> 
<img src="tasks/example_tasks/images/3.2.1.png" width="250"/> <img src="tasks/example_tasks/images/3.2.2.png" width="250"/> <img src="tasks/example_tasks/images/3.2.3.png" width="250"/> 
<img src="tasks/example_tasks/images/3.2.4.png" width="250"/> <img src="tasks/example_tasks/images/3.3.1.png" width="250"/> <img src="tasks/example_tasks/images/3.3.2.png" width="250"/> 
<img src="tasks/example_tasks/images/3.3.3.png" width="250"/> <img src="tasks/example_tasks/images/3.3.4.png" width="250"/> <img src="tasks/example_tasks/images/3.4.1.png" width="250"/> 
<img src="tasks/example_tasks/images/3.4.2.png" width="250"/> <img src="tasks/example_tasks/images/3.4.3.png" width="250"/> <img src="tasks/example_tasks/images/3.4.4.png" width="250"/> 
<img src="tasks/example_tasks/images/3.5.1.png" width="250"/> <img src="tasks/example_tasks/images/3.5.2.png" width="250"/> <img src="tasks/example_tasks/images/3.5.3.png" width="250"/> 
<img src="tasks/example_tasks/images/3.5.4.png" width="250"/> <img src="tasks/example_tasks/images/3.5.5.png" width="250"/> <img src="tasks/example_tasks/images/3.6.1.png" width="250"/> 
<img src="tasks/example_tasks/images/3.6.2.png" width="250"/> <img src="tasks/example_tasks/images/3.6.3.png" width="250"/> <img src="tasks/example_tasks/images/3.6.4.png" width="250"/> 
<img src="tasks/example_tasks/images/3.6.5.png" width="250"/> <img src="tasks/example_tasks/images/3.7.1.png" width="250"/> <img src="tasks/example_tasks/images/3.7.2.png" width="250"/> 
<img src="tasks/example_tasks/images/3.7.3.png" width="250"/> <img src="tasks/example_tasks/images/3.7.4.png" width="250"/> <img src="tasks/example_tasks/images/3.7.5.png" width="250"/> 
<img src="tasks/example_tasks/images/3.8.1.png" width="250"/> <img src="tasks/example_tasks/images/3.8.2.png" width="250"/> <img src="tasks/example_tasks/images/3.9.1.png" width="250"/> 
<img src="tasks/example_tasks/images/3.9.2.png" width="250"/> <img src="tasks/example_tasks/images/3.9.3.png" width="250"/> <img src="tasks/example_tasks/images/3.9.4.png" width="250"/> 
<img src="tasks/example_tasks/images/3.9.5.png" width="250"/> <img src="tasks/example_tasks/images/3.9.6.png" width="250"/>

## 3. Task generator
<a name="Task-generator"/></a>
We develop a task generator that can generate tasks for the task templates we designed.<br>
1. To run the task generator:<br>
    1. Go to ```tasks/task_generator```
    2. Copy the task templates that you want to generate tasks into the ```input``` (level templates can be found in ```tasks/task_templates```)
    3. Run the tak generator providing the number of tasks as an argument
     ```
        python generate_tasks.py <number of tasks to generate>
     ```
    4. Generated tasks will be available in the ```output```

## 4. Tasks generated for baseline analysis
<a name="Tasks-generated-for-baselines"/></a>
We generated 100 tasks from each of the 65 task templates for the baseline analysis. The generated tasks can be found in ```tasks/generated_tasks.zip```. After extracting this file, the generatd tasks can be found following the folder structure: <br>
&nbsp;&nbsp;&nbsp;&nbsp;generated_tasks/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- index of the hierarchy level/ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- index of the capability/ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- index of the template/ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- task files with the indexes of the tasks <br>	

## 5. Baseline Agents and the Framework<a name="BAF"></a>

Tested environments:
- Ubuntu: 18.04/20.04
- Python: 3.9
- Numpy: 1.20
- torch: 1.8.1
- torchvision: 0.9.1
- lxml: 4.6.3
- tensorboard: 2.5.0
- Java: 13.0.2/13.0.7

Before running agents, please:

1. Go to ```buildgame``` and unzip ```Linux.zip```
2. Go to ```task/generated_tasks``` and unzip ```generated_tasks.zip```

### 5.1 How to run heuristic agents<a name="RHA"></a>

1. Run Java heuristic agents: Datalab and Eagle Wings: <br>

    1. Go to ```Utils``` and in terminal run
          ```
          python PrepareTestConfig.py
          ```
    2. Go to ```buildgame/Linux```, in terminal run
          ```sh
          java -jar game_playing_interface.jar
          ```
    3. Go to ```Agents/HeuristicAgents/``` and in terminal run Datalab
        ```sh
        java -jar datalab_037_v4_java12.jar 1
        ```
       or Eagle Wings
          ```sh
          java -jar eaglewings_037_v3_java12.jar 1
          ```
2. Run *Random Agent* and *Pig Shooter*: <br>
    1. Go to ```Agents/```
    2. In terminal, after grant execution permission run Random Agent
       ```sh
       ./TestPythonHeuristicAgent.sh RandomAgent
       ```
       or Pig Shooter
       ```sh
       ./TestPythonHeuristicAgent.sh PigShooter
       ```

### 5.2 How to run DQN Baseline<a name="RLA"></a>

1. Go to ```Agents/```
2. In terminal, after grant execution permission, train the agent for within capability training
    ```sh
    ./TrainLearningAgent.sh within_capability
    ```
   and for within template training
    ```sh
    ./TrainLearningAgent.sh within_template
    ```
3. Models will be saved to ```Agents/LearningAgents/saved_model```
4. To test learning agents, go the folder ```Agents```:
    1. test within template performance, run
    ```
    python TestAgentOfflineWithinTemplate.py
    ```
    2. test within capability performance, run
    ```
    python TestAgentOfflineWithinCapability.py
    ```

### 5.3 How to develop your own agent <a name="ROA"></a>

We provide a gym-like environment. For a simple demo, which can be found at ```demo.py```

```python
from SBAgent import SBAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper

# for using reward as score and 50 times faster game play
env = SBEnvironmentWrapper(reward_type="score", speed=50)
level_list = [1, 2, 3]  # level list for the agent to play
dummy_agent = SBAgent(env=env, level_list=level_list)  # initialise agent
dummy_agent.state_representation_type = 'image'  # use symbolic representation as state and headless mode
env.make(agent=dummy_agent, start_level=dummy_agent.level_list[0],
         state_representation_type=dummy_agent.state_representation_type)  # initialise the environment

s, r, is_done, info = env.reset()  # get ready for running
for level_idx in level_list:
    is_done = False
    while not is_done:
        s, r, is_done, info = env.step([-100, -100])  # agent always shoots at -100,100 as relative to the slingshot

    env.current_level = level_idx+1  # update the level list once finished the level
    if env.current_level > level_list[-1]: # end the game when all game levels in the level list are played
        break
    s, r, is_done, info = env.reload_current_level() #go to the next level
```
    
### 5.4 Outline of the Agent Code <a name="Code"></a>

The ```./Agents``` folder contains all the relevant source code of our agents. Below is the outline of the code (this is a
simple description. Detailed documentation in progress):

1. ```Client```:
    1. ```agent_client.py```: Includes all communication protocols.
2. ```final_run```: Place to store tensor board results.
3. ```HeuristicAgents```
    1. ```datalab_037_v4_java12.jar```: State-of-the-art java agent for Angry Birds.
    2. ```eaglewings_037_v3_java12.jar```: State-of-the-art java agent for Angry Birds.
    3. ```PigShooter.py```: Python agent that shoots at the pigs only.
    4. ```RandomAgent.py```: Random agent that choose to shoot from $x \in (-100,-10)$ and $y \in (-100,100)$.
    5. ```HeuristicAgentThread.py```: A thread wrapper to run multi-instances of heuristic agents.
4. ```LearningAgents```
    1. ```RLNetwork```: Folder includes all DQN structures that can be used as an input to ```DQNDiscreteAgent.py```.
    2. ```saved_model```: Place to save trained models.
    3. ```LearningAgent.py```: Inherited from SBAgent class, a base class to implement learning agents.
    4. ```DQNDiscreteAgent.py```: Inherited from LearningAgent, a DQN agent that has discrete action space.
    5. ```LearningAgentThread.py```: A thread wrapper to run multi-instances of learning agents.
    6. ```Memory.py```: A script that includes different types of memories. Currently, we have normal memory,
       PrioritizedReplayMemory and PrioritizedReplayMemory with balanced samples.
5. ```SBEnvironment```
    1. ```SBEnvironmentWrapper.py```: A wrapper class to provide gym-like environment.
6. ```StateReader```: Folder that contains files to convert symbolic state representation to inputs to the agents.
7. ```Utils```:
    1. ```Config.py```: Config class that used to pass parameter to agents.
    2. ```GenerateCapabilityName.py```: Generate a list of names of capability for agents to train.
    3. ```GenerateTemplateName.py```: Generate a list of names of templates for agents to train.
    4. ```LevelSelection.py```: Class that includes different strategies to select levels. For example, an agent may
       choose to go to the next level if it passes the current one, or only when it has played the current level for a
       predefined number of times.
    5. ```NDSparseMatrix.py```: Class to store converted symbolic representation in a sparse matrix to save memory
       usage.
    6. ```Parameters.py```: Training/testing parameters used to pass to the agent.
    7. ```PrepareTestConfig.py```: Script to generate config file for the game console to use for testing agents only.
    8. ```trajectory_planner.py```:  It calculates two possible trajectories given a directly reachable target point. It
       returns None if the target is non-reachable by the bird
8. ```demo.py```: A demo to showcase how to use the framework.
9. ```SBAgent.py```: Base class for all agents.
10. ```MultiAgentTestOnly.py```: To test python heuristic agents with running multiple instances on one particular template.
11. ```TestAgentOfflineWithinCapability.py```: Using the saved models in ```LearningAgents/saved_model``` to test agent's
    within capability performance on test set.
12. ```TestAgentOfflineWithinTemplate.py```: Using the saved models in ```LearningAgents/saved_model``` to test agent's
    within template performance on test set.
13. ```TrainLearningAgent.py```: Script to train learning agents on particular template with defined mode.
14. ```TestPythonHeuristicAgent.sh```: Bash Script to test heuristic agent's performance on all templates.
15. ```TrainLearningAgent.sh```: Bash Script to train learning agents on all templates/capabilities. 

## 6. Framework<a name="Framework"></a>

### 6.1 The Game Environment<a name="Env"></a>

1. The coordination system
    - in the science birds game, the origin point (0,0) is the bottom-left corner, and the Y coordinate increases along
      the upwards direction, otherwise the same as above.
    - Coordinates ranging from (0,0) to (640,480).

### 6.2 Symbolic Representation Data Structure<a name="SymbolicRepresentation"></a>

1. Symbolic Representation data of game objects is stored in a Json object. The json object describes an array where each element
   describes a game object. Game object categories, and their properties are described below:
    - Ground: the lowest unbreakable flat support surface
        - property: id = 'object [i]'
        - property: type = 'Ground'
        - property: yindex = [the y coordinate of the ground line]
    - Platform: Unbreakable obstacles
        - property: id = 'object [i]'
        - property: type = 'Object'
        - property: vertices = [a list of ordered 2d points that represents the polygon shape of the object]
        - property: colormap = [a list of compressed 8-bit (RRRGGGBB) colour and their percentage in the object]
    - Trajectory: the dots that represent the trajectories of the birds
        - property: id = 'object [i]'
        - property: type = 'Trajectory'
        - property: location = [a list of 2d points that represents the trajectory dots]

    - Slingshot: Unbreakable slingshot for shooting the bird
        - property: id = 'object [i]'
        - property: type = 'Slingshot'
        - property: vertices = [a list of ordered 2d points that represents the polygon shape of the object]
        - property: colormap = [a list of compressed 8-bit (RRRGGGBB) colour and their percentage in the object]
    - Red Bird:
        - property: id = 'object [i]'
        - property: type = 'Object'
        - property: vertices = [a list of ordered 2d points that represents the polygon shape of the object]
        - property: colormap = [a list of compressed 8-bit (RRRGGGBB) colour and their percentage in the object]
    - all objects below have the same representation as red bird
    - Blue Bird:
    - Yellow Bird:
    - White Bird:
    - Black Bird:
    - Small Pig:
    - Medium Pig:
    - Big Pig:
    - TNT: an explosive block
    - Wood Block: Breakable wooden blocks
    - Ice Block: Breakable ice blocks
    - Stone Block: Breakable stone blocks
   
2. Round objects are also represented as polygons with a list of vertices
3. Symbolic Representation with noise
    - If noisy Symbolic Representation is requested, the noise will be applied to each point in vertices of the game objects except
      the **ground**, **all birds** and the **slingshot**
    - The noise for 'vertices' is applied to all vertices with the same amount within 5 pixels
    - The colour map has a noise of +/- 2%.
    - The colour is the colour map compresses 24 bit RGB colour into 8 bit
        - 3 bits for Red, 3 bits for Green and 2 bits for Blue
        - the percentage of the colour that accounts for the object is followed by colour
        - example: (127, 0.5) means 50% pixels in the objects are with colour 127
    - The noise is uniformly distributed
    - We will later offer more sophisticated and adjustable noise.

### 6.3 Communication Protocols<a name="Protocol"></a>

<table style="text-align:center;">
    <thead>
        <tr>
            <th>Message ID</th>
            <th>Request</th>
            <th>Format (byte[ ])</th>
			<th>Return</th>
			<th>Format (byte[ ])</th>
        </tr>
    </thead>
    <tbody>
		<tr>
			<td>1-10</td>
			<td colspan=4>Configuration Messages</td>			
		</tr>	
		<tr>
			<td>1</td>
			<td>Configure team ID<br /> Configure running mode</td>
			<td>[1][ID][Mode]<br />ID: 4 bytes<br />Mode: 1 byte<br/>
			COMPETITION = 0<br/>TRAINING = 1</td>
			<td> Four bytes array.<br />
			The first byte indicates the round;<br />
			the second specifies the time limit in minutes;<br />
			the third specifies the number of available levels<br /></td>
			<td>[round info][time limit][available levels]<br />
			Note: in training mode, the return will be [0][0][0].<br />
			As the round info is not used in training,<br />
			the time limit will be 600 hours, <br />
			and the number of levels needs to be requested via message ID 15 
			</td>	
		</tr>	
		<tr>
			<td>2</td>
			<td>Set simulation speed<br />speed$\in$[0.0, 50.0]
			<br />Note: this command can be sent at anytime during playing to change the simulation speed</td>
			<td>[2][speed]<br />speed: 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>	
		</tr>	
		<tr>
			<td>11-30</td>
			<td colspan=4>Query Messages</td>		
		</tr>
		<tr>
			<td>11</td>
			<td>Do Screenshot</td>
			<td>[11]</td>
			<td>Width, height, image bytes<br/>
			Note: this command only returns screenshots without symbolic representation </td>
			<td>[width][height][image bytes]<br />width, height: 4 bytes</td>
		</tr>
		<tr>
			<td>12</td>
			<td>Get game state</td>
			<td>[12]</td>
			<td>One byte indicates the ordinal of the state</td>
			<td>[0]: UNKNOWN<br />
			[1] : MAIN_MENU<br />
			[2]: EPISODE_MENU<br />
			[3]: LEVEL_SELECTION<br />
			[4]: LOADING<br />
			[5]: PLAYING<br />
			[6]: WON<br />
			[7]: LOST</td>
		</tr>
		<tr>
			<td>14</td>
			<td>Get the current level</td>
			<td>[14]</td>
			<td>four bytes array indicates the index of the current level</td>
			<td>[level index]
		</tr>
		<tr>
			<td>15</td>
			<td>Get the number of levels</td>
			<td>[15]</td>
			<td>four bytes array indicates the number of available levels</td>
			<td>[number of level]</td>
		</tr>
		<tr>
			<td>23</td>
			<td>Get my score</td>
			<td>[23]</td>
			<td>A 4 bytes array indicating the number of levels <br/> followed by ([number_of_levels] * 4) bytes array with every four<br/> slots indicates a best score for the corresponding level</td>
			<td>[number_of_levels][score_level_1]....[score_level_n]<br/>
			Note: This should be used carefully for the training mode, <br/>
			because there may be large amount of levels used in the training.<br/>
			Instead, when the agent is in winning state,<br/>
			use message ID 65 to get the score of a single level at winning state</td>
		</tr>
		<tr>
			<td>31-50</td>
			<td colspan=4>In-Game Action Messages</td>		
		</tr>
		<tr>
			<td>31</td>
			<td>Shoot using the Cartesian coordinates [Safe mode*]<br\>
			</td>
			<td>[31][fx][fy][dx][dy][t1][t2]<br/>
			focus_x : the x coordinate of the focus point<br/>
			focus_y: the y coordinate of the focus point<br/>
			dx: the x coordinate of the release point minus focus_x<br/>
			dy: the y coordinate of the release point minus focus_y<br/>
			t1: the release time<br/>
			t2: the gap between the release time and the tap time.<br/>
			If t1 is set to 0, the server will execute the shot immediately.<br/>
			The length of each parameter is 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>32</td>
			<td>Shoot using Polar coordinates [Safe mode*]</td>
			<td>[32][fx][fy][theta][r][t1][t2]<br/>
			theta: release angle<br/>
			r: the radial coordinate<br/>
			The length of each parameter is 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>33</td>
			<td>Sequence of shots [Safe mode*]</td>
			<td>[33][shots length][shot message ID][Params]...[shot message ID][Params]<br/>
			Maximum sequence length: 16 shots</td>
			<td>An array with each slot indicates good/bad shot.<br/>
			The bad shots are those shots that are rejected by the server</td>
			<td>For example, the server received 5 shots, and the third one<br/> 
			was not executed due to some reason, then the server will return<br/>
			[1][1][0][1][1]</td>
		</tr>
		<tr>
			<td>41</td>
			<td>Shoot using the Cartesian coordinates [Fast mode**]<br\>
			</td>
			<td>[41][fx][fy][dx][dy][t1][t2]<br/>
			The length of each parameter is 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>42</td>
			<td>Shoot using Polar coordinates [Fast mode**]</td>
			<td>[42][fx][fy][theta][r][t1][t2]<br/>
			The length of each parameter is 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>43</td>
			<td>Sequence of shots [Fast mode**]</td>
			<td>[43][shots length][shot message ID][Params]...[shot message ID][Params]<br/>
			Maximum sequence length: 16 shots</td>
			<td>An array with each slot indicates good/bad shot.<br/>
			The bad shots are those shots that are rejected by the server</td>
			<td>For example, the server received 5 shots, and the third one<br/> 
			was not executed due to some reason, then the server will return<br/>
			[1][1][0][1][1]</td>
		</tr>
		<tr>
			<td>34</td>
			<td>Fully Zoom Out</td>
			<td>[34]</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>35</td>
			<td>Fully Zoom In</td>
			<td>[35]</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>51-60</td>
			<td colspan=4>Level Selection Messages</td>		
		</tr>
		<tr>
			<td>51</td>
			<td>Load a level</td>
			<td>[51][Level]<br/>
			Level: 4 bytes</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>52</td>
			<td>Restart a level</td>
			<td>[52]</td>
			<td>OK/ERR</td>
			<td>[1]/[0]</td>
		</tr>
		<tr>
			<td>61-70</td>
			<td colspan=4>Science Birds Specific Messages</td>		
		</tr>
		<tr>
			<td>61</td>
			<td>Get Symbolic Representation With Screenshot</td>
			<td>[61]</td>
			<td>Symbolic Representation and corresponding screenshot</td>
			<td>[symbolic representation byte array length][Symbolic Representation bytes][image width][image height][image bytes]<br/>
			symbolic representation byte array length: 4 bytes<br/>
			image width: 4 bytes
			image height: 4 bytes</td>
		</tr>
		<tr>
			<td>62</td>
			<td>Get Symbolic Representation Without Screenshot</td>
			<td>[62]</td>
			<td>Symbolic Representation</td>
			<td>[symbolic representation byte array length][Symbolic Representation bytes]</td>
		</tr>
		<tr>
			<td>63</td>
			<td>Get Noisy Symbolic Representation With Screenshot</td>
			<td>[63]</td>
			<td>noisy Symbolic Representation and corresponding screenshot</td>
			<td>[symbolic representation byte array length][Symbolic Representation bytes][image width][image height][image bytes]</td>
		</tr>
		<tr>
			<td>64</td>
			<td>Get Noisy Symbolic Representation Without Screenshot</td>
			<td>[64]</td>
			<td>noisy Symbolic Representation</td>
			<td>[symbolic representation byte array length][Symbolic Representation bytes]</td></tr>
		<tr>
			<td>65</td>
			<td>Get Current Level Score</td>
			<td>[65]</td>
			<td>current score<br/>
			Note: this score can be requested at any time at Playing/Won/Lost state<br/>
			This is used for agents that take intermediate score seriously during training/reasoning<br/>
			To get the winning score, please make sure to execute this command when the game state is "WON"</td>
			<td>[score]<br/>
			score: 4 bytes</td>
		</tr>
		<tr>
			<td colspan=5>* Safe Mode: The server will wait until the state is static after making a shot.</td>
		</tr>
		<tr>
			<td colspan=5>** Fast mode: The server will send back a confirmation once a shot is made. 
			The server will not do any check for the appearance of the won page.</td>
		</tr>

	</tbody>

</table>

