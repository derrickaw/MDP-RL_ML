//package burlap.tutorials.bpl;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.common.SinglePFTF;
import burlap.oomdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.environment.Environment;
import burlap.oomdp.singleagent.environment.EnvironmentServer;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.singleagent.MDPSolver;
import burlap.oomdp.statehashing.HashableState;
import burlap.behavior.valuefunction.QValue;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.behavior.valuefunction.QFunction;




import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.*;
import java.io.PrintWriter;
import java.io.File;
import java.io.FileNotFoundException;


public class rlVIPI{ //extends MDPSolver implements QFunction{

	GridWorldDomain 				gw;
	Domain 							domain;
	RewardFunction					rf;
	TerminalFunction				tf;
	StateConditionTest				goalCondition;
	State 							initialState;
	HashableStateFactory			hashingFactory;
	Environment						env;
	Policy 							learningPolicy;
	//Map<HashableState, List<QValue>> qValues;
	//ValueFunctionInitialization qinit;


	public static void main(String [] args) throws FileNotFoundException{
		rlVIPI example = new rlVIPI();
		List<Double> rewards = Arrays.asList(0.0);
		List<Double> penalties = Arrays.asList(-0.1);
		List<Double> transitions = Arrays.asList(0.8);
		List<Double> discounts = Arrays.asList(0.99);
		double precision = 0.001;
		int maxIterations = 1000;
		String outputPath = "output/";
		boolean vis = false;
		int count = 0;


		// 4-Room - VI
		for(Double reward : rewards){
			for(Double penalty : penalties){
				for(Double transition : transitions){
					for(Double discount : discounts){
						System.out.format("4Room - Value: reward, penalty, transition, " +
							"discount: %f %f %f %f\n", reward, penalty, transition, discount);
						example.valueIterationGrid(true, 11, 11, 10, 10, transition, reward, 
							penalty, discount, precision, maxIterations, outputPath, count,
							vis);
					}
				}
			}
		}

		// Long Hallway - PI
		// count = 0;
		// for(Double reward : rewards){
		// 	for(Double penalty : penalties){
		// 		for(Double transition : transitions){
		// 			for(Double discount : discounts){
		// 				System.out.format("Long Hallway - Value: reward, penalty, transition, " +
		// 					"discount: %f %f %f %f\n", reward, penalty, transition, discount);
		// 				example.valueIterationGrid(false, 170, 3, 169, 1, transition, reward, 
		// 					penalty, discount, precision, maxIterations, outputPath, count,
		// 					vis);
		// 				count++;
		// 			}
		// 		}
		// 	}
		// }



		// 4-Room - PI
		// count = 0;
		for(Double reward : rewards){
			for(Double penalty : penalties){
				for(Double transition : transitions){
					for(Double discount : discounts){
						System.out.format("4Room - Policy: reward, penalty, transition, " +
							"discount: %f %f %f %f\n", reward, penalty, transition, discount);
						example.policyIterationGrid(true, 11, 11, 10, 10, transition, reward, 
							penalty, discount, precision, maxIterations, outputPath, count,
							vis);
						//count++;
					}
				}
			}
		}

		// Long Hallway - PI
		// count = 0;
		// for(Double reward : rewards){
		// 	for(Double penalty : penalties){
		// 		for(Double transition : transitions){
		// 			for(Double discount : discounts){
		// 				System.out.format("Long Hallway - Policy: reward, penalty, transition, " +
		// 					"discount: %f %f %f %f\n", reward, penalty, transition, discount);
		// 				example.policyIterationGrid(false, 170, 3, 169, 1, transition, reward, 
		// 					penalty, discount, precision, maxIterations, outputPath, count,
		// 					vis);
		// 				count++;
		// 			}
		// 		}
		// 	}
		// }


		//writer.close();



		//example.valueIterationGrid();
		//example.policyIterationGrid();
		//example.qLearningGridHall();

	}

	public rlVIPI(){
		//this.hashingFactory = new SimpleHashableStateFactory();
		//QFunction planner = new QFunction()
		//this.learningPolicy = new EpsilonGreedy(hello, 0.1);
		//this.qValues = new HashMap<HashableState, List<QValue>>();

	}



	public void valueIterationGrid(boolean rooms, int width, int height, int goalX, int goalY, 
		double trans, double reward, double penalty, double discount, double precision, 
		int maxIterations, String outputPath, int count, boolean vis){

		// Set domain
		gw = new GridWorldDomain(width, height); 
		if (rooms){
			gw.setMapToFourRooms();
		}
		gw.setProbSucceedTransitionDynamics(trans); //stochastic transitions with 0.8 success rate
		domain = gw.generateDomain(); //generate the grid world domain
		
		//setup initial state
		initialState = GridWorldDomain.getOneAgentNLocationState(domain, 1);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, goalX, goalY);

		//ends when the agent reaches a location
		tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));

		//reward function definition
		rf = new GoalBasedRF(new TFGoalCondition(tf), reward, penalty);

		hashingFactory = new SimpleHashableStateFactory();
		env = new SimulatedEnvironment(domain, rf, tf, initialState);

		Planner planner = new ValueIteration(domain, rf, tf, discount, hashingFactory, 
			precision, maxIterations);
		double beginTime = System.currentTimeMillis();
		Policy p = planner.planFromState(initialState);
		double endTime = System.currentTimeMillis();
		System.out.printf("Time: %.2f",endTime-beginTime);
		System.out.println();
		//System.out.println(p.getActionDistributionForState(initialState));


		outputPath += "vi" + Integer.toString(count);
		//p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath);

		//simpleValueFunctionVis((ValueFunction)planner, p, initialState, hashingFactory, domain);
		//manualValueFunctionVis((ValueFunction)planner, p, s, hashingFactory, domain);

	}

	public void policyIterationGrid(boolean rooms, int width, int height, int goalX, int goalY, 
		double trans, double reward, double penalty, double discount, double precision, 
		int maxIterations, String outputPath, int count, boolean vis){

		gw = new GridWorldDomain(width,height); 
		if (rooms){
			gw.setMapToFourRooms();
		}
		gw.setProbSucceedTransitionDynamics(trans); //stochastic transitions with 0.8 success rate
		domain = gw.generateDomain(); //generate the grid world domain		

		//setup initial state
		initialState = GridWorldDomain.getOneAgentNLocationState(domain, 1);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, goalX, goalY);

		//ends when the agent reaches a location
		tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));

		//reward function definition
		rf = new GoalBasedRF(new TFGoalCondition(tf), reward, penalty);

		hashingFactory = new SimpleHashableStateFactory();
		env = new SimulatedEnvironment(domain, rf, tf, initialState);


		Planner planner = new PolicyIteration(domain, rf, tf, discount, hashingFactory, 
			precision, precision, maxIterations, maxIterations);
		double beginTime = System.currentTimeMillis();
		Policy p = planner.planFromState(initialState);
		double endTime = System.currentTimeMillis();
		System.out.printf("Time: %.2f",endTime-beginTime);
		System.out.println();

		outputPath += "pi" + Integer.toString(count);
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath);

		//simpleValueFunctionVis((ValueFunction)planner, p, initialState, hashingFactory, domain);
		//manualValueFunctionVis((ValueFunction)planner, p);

	}


	public static void simpleValueFunctionVis(ValueFunction valueFunction, Policy p, State s, 
		HashableStateFactory hashingFactory, Domain domain){

		List<State> allStates = StateReachability.getReachableStates(s, (SADomain)domain, hashingFactory);
		// for (State state : allStates){
		// 	System.out.println(p.getActionDistributionForState(state));
		// }
		// System.out.println(p);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, valueFunction, p);
		gui.initGUI();

	}

	public static void manualValueFunctionVis(ValueFunction valueFunction, Policy p, State s, 
		HashableStateFactory hashingFactory, Domain domain){

		List<State> allStates = StateReachability.getReachableStates(s, (SADomain)domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, specifying which attributes correspond to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
				GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);


		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
				GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);

		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();



	}


	// @Override
	// public void resetSolver() {
	// 	this.qValues.clear();
	// }

	// @Override
	// public List<QValue> getQs(State s) {
	// 	//first get hashed state
	// 	HashableState sh = this.hashingFactory.hashState(s);

	// 	//check if we already have stored values
	// 	List<QValue> qs = this.qValues.get(sh);

	// 	//create and add initialized Q-values if we don't have them stored for this state
	// 	if(qs == null){
	// 		List<GroundedAction> actions = this.getAllGroundedActions(s);
	// 		qs = new ArrayList<QValue>(actions.size());
	// 		//create a Q-value for each action
	// 		for(GroundedAction ga : actions){
	// 			//add q with initialized value
	// 			qs.add(new QValue(s, ga, this.qinit.qValue(s, ga)));
	// 		}
	// 		//store this for later
	// 		this.qValues.put(sh, qs);
	// 	}

	// 	return qs;
	// }

	// @Override
	// public QValue getQ(State s, AbstractGroundedAction a) {
	// 	//first get all Q-values
	// 	List<QValue> qs = this.getQs(s);

	// 	//translate action parameters to source state for Q-values if needed
	// 	a = ((GroundedAction)a).translateParameters(s, qs.get(0).s);

	// 	//iterate through stored Q-values to find a match for the input action
	// 	for(QValue q : qs){
	// 		if(q.a.equals(a)){
	// 			return q;
	// 		}
	// 	}

	// 	throw new RuntimeException("Could not find matching Q-value.");
	// }

	// @Override
	// public double value(State s) {
	// 	return QFunctionHelper.getOptimalValue(this, s);
	// }




}
