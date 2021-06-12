## Baseline Agents
Before running agents, please:
   1. Go to ```buildgame``` and unzip ```Linux.zip```
   2. Go to ```level_varitions/generated_levels``` and unzip ```fourth generation.zip```
### How to run heuristic agents
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
2.  Run *Random Agent* and *Pig Shooter*: <br>
    1. Go to Agents/
    2. In terminal, after grant execution permission run Random Agent
       ```sh
       ./TestPythonHeuristicAgent.sh RandomAgent
       ```
       or Pig Shooter
       ```sh
       ./TestPythonHeuristicAgent.sh PigShooter
       ```
### How to run DQN
 1. Go to Agents/
 2. In terminal, after grant execution permission, train the agent by running 
    ```sh
    ./TrainLearningAgent.sh within_capability
    ```
    for within capability
    and
    ```sh
    ./TrainLearningAgent.sh within_template
    ```
    for within capability
    or Pig Shooter
    ```sh
    ./TestPythonHeuristicAgent.sh PigShooter
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
### How to develop your own agent

4. How the framework works
5. science birds API