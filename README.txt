CS4641 - HW4 - MDP and Reinforcement Learning
By: Derrick Williams
Date: 2016-04-24

Data package:
BURLAP was used to implement this project.  "The Brown-UMBC Reinforcement Learning and Planning (BURLAP) java code library is for the use and development of single or multi-agent planning and learning algorithms and domains to accompany them." - James MacGlashan

Running Code:
1. A testcode folder is provided in the Code folder that has the a lib folder contained within it that contains the burlap.jar file and all the necessary dependences.  Also within the testcode folder are the two java files for running value and policy iteration and Q-Learning algorithm.  The testcode folder should be placed somewhere on your network where you want to run the code such as your desktop.
2. Navigate now to this location Desktop/testcode ('cd testcode', if you are currently in the Desktop folder and you actually placed the testcode folder on the Desktop).
3. Type 'javac -cp lib/*:. rlVIPI.java' to compile the value and policy iteration java file.
4. Type 'java -cp lib/*:. rlVIPI' to run the code.  You will have to uncomment certain sections of code to run each set of iterations for both MDP and value and policy iterations.  Also uncomment if you want simulations.
5. Type 'javac -cp lib/*:. rlQL.java' to compile the value and policy iteration java file.
6. Type 'java -cp lib/*:. rlQL' to run the code.  You will have to change the size of the grid and reward location for each MDP problem for Q-Learning.  Also uncomment if you want performance simulations.


Code Folder:
testcode						- Folder with lib folder with java packages for ML algorithms and java files
testcode/lib 					- Folder with java packages for implementing BURLAP ML algorithms
rlVIPI.java 					- File to compile and run value and policy iteration for each MDP
rlQL.java 						- File to compile and run Q-Learning for each MDP

Results Folder:
4RVI.png 						- Four Room MDP problem policy output for value iteration
4RPI.png 						- Four Room MDP problem policy output for policy iteration
4RQL.png 						- Four Room MDP problem policy output for Q-Learning 
4RQL-01_01 - 09_09 				- Four Room MDP problem Q-Learning performance output
LHVI.png 						- Long Hallway MDP problem policy output for value iteration
LHPI.png 						- Long Hallway MDP problem policy output for policy iteration
LHQLa.png 						- Long Hallway MDP problem policy output for Q-Learning
LHQLa_01_01 - 09_07 			- Long Hallway MDP problem Q-Learning performance output
rl-Results.xlsx					- results from all MDP problem runs
