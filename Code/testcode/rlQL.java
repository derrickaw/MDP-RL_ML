//package burlap.tutorials.cpl;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.environment.Environment;
import burlap.oomdp.singleagent.environment.EnvironmentOutcome;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableState;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import burlap.oomdp.auxiliary.common.SinglePFTF;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.oomdp.singleagent.SADomain;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.behavior.singleagent.learning.LearningAgentFactory;




import java.util.*;


public class rlQL extends MDPSolver implements LearningAgent, QFunction {

	Map<HashableState, List<QValue>> qValues;
	ValueFunctionInitialization qinit;
	double learningRate;
	Policy learningPolicy;

	public rlQL(Domain domain, double gamma, HashableStateFactory hashingFactory,
					  ValueFunctionInitialization qinit, double learningRate,  double epsilon){

		this.solverInit(domain, null, null, gamma, hashingFactory);
		this.qinit = qinit;
		this.learningRate = learningRate;
		this.qValues = new HashMap<HashableState, List<QValue>>();
		this.learningPolicy = new EpsilonGreedy(this, epsilon);

	}

	@Override
	public EpisodeAnalysis runLearningEpisode(Environment env) {
		return this.runLearningEpisode(env, -1);
	}

	@Override
	public EpisodeAnalysis runLearningEpisode(Environment env, int maxSteps) {
		//initialize our episode analysis object with the initial state of the environment
		EpisodeAnalysis ea = new EpisodeAnalysis(env.getCurrentObservation());

		//behave until a terminal state or max steps is reached
		State curState = env.getCurrentObservation();
		int steps = 0;
		while(!env.isInTerminalState() && (steps < maxSteps || maxSteps == -1)){

			//select an action
			GroundedAction a = (GroundedAction)this.learningPolicy.getAction(curState);

			//take the action and observe outcome
			EnvironmentOutcome eo = a.executeIn(env);

			//record result
			ea.recordTransitionTo(a, eo.op, eo.r);

			//get the max Q value of the resulting state if it's not terminal, 0 otherwise
			double maxQ = eo.terminated ? 0. : this.value(eo.op);

			//update the old Q-value
			QValue oldQ = this.getQ(curState, a);
			oldQ.q = oldQ.q + this.learningRate * (eo.r + this.gamma * maxQ - oldQ.q);


			//move on to next state
			curState = eo.op;
			steps++;

		}

		return ea;
	}

	@Override
	public void resetSolver() {
		this.qValues.clear();
	}

	@Override
	public List<QValue> getQs(State s) {
		//first get hashed state
		HashableState sh = this.hashingFactory.hashState(s);

		//check if we already have stored values
		List<QValue> qs = this.qValues.get(sh);

		//create and add initialized Q-values if we don't have them stored for this state
		if(qs == null){
			List<GroundedAction> actions = this.getAllGroundedActions(s);
			qs = new ArrayList<QValue>(actions.size());
			//create a Q-value for each action
			for(GroundedAction ga : actions){
				//add q with initialized value
				qs.add(new QValue(s, ga, this.qinit.qValue(s, ga)));
			}
			//store this for later
			this.qValues.put(sh, qs);
		}

		return qs;
	}

	@Override
	public QValue getQ(State s, AbstractGroundedAction a) {
		//first get all Q-values
		List<QValue> qs = this.getQs(s);

		//translate action parameters to source state for Q-values if needed
		a = ((GroundedAction)a).translateParameters(s, qs.get(0).s);

		//iterate through stored Q-values to find a match for the input action
		for(QValue q : qs){
			if(q.a.equals(a)){
				return q;
			}
		}

		throw new RuntimeException("Could not find matching Q-value.");
	}

	@Override
	public double value(State s) {
		return QFunctionHelper.getOptimalValue(this, s);
	}

	public static void simpleValueFunctionVis(ValueFunction valueFunction, Policy p, State initialState, 
		HashableStateFactory hashingFactory, Domain domain){

		List<State> allStates = StateReachability.getReachableStates(initialState, (SADomain)domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, valueFunction, p);
		gui.initGUI();

	}

	public static void qLearnGrid(boolean rooms, int width, int height, int goalX, int goalY, 
		double trans, double reward, double penalty, double discount, double learningRate, 
		double epsilon, double precision, int maxIterations, String outputPath, int count, 
		boolean vis){

		GridWorldDomain gwd = new GridWorldDomain(width, height);
		if (rooms){
			gwd.setMapToFourRooms();
		}
		gwd.setProbSucceedTransitionDynamics(trans);

		Domain domain = gwd.generateDomain();

		//get initial state with agent in 0,0
		//State s = GridWorldDomain.getOneAgentNoLocationState(domain);
		State s = GridWorldDomain.getOneAgentNLocationState(domain, 1);
		GridWorldDomain.setAgent(s, 0, 0);
		GridWorldDomain.setLocation(s, 0, goalX, goalY);

		//terminate in top right corner
		//TerminalFunction tf = new GridWorldTerminalFunction(0, 9);
		TerminalFunction tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));

		//all transitions return -1
		//RewardFunction rf = new UniformCostRF();
		RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), reward, penalty);

		//create environment
		SimulatedEnvironment env = new SimulatedEnvironment(domain,rf, tf, s);

		//create Q-learning
		rlQL agent = new rlQL(domain, discount, new SimpleHashableStateFactory(),
				new ValueFunctionInitialization.ConstantValueFunctionInitialization(), 
				learningRate, epsilon);

		//run Q-learning and store results in a list
		List<EpisodeAnalysis> episodes = new ArrayList<EpisodeAnalysis>(maxIterations);
		HashableStateFactory hashingFactory = new SimpleHashableStateFactory();
		

		double beginTime = System.currentTimeMillis();

		for(int i = 0; i < maxIterations; i++){
			EpisodeAnalysis ea = agent.runLearningEpisode(env);
			episodes.add(ea);
			System.out.println(i + ": " + ea.maxTimeStep());


			//ea.writeToFile(outputPath + "ql_" + i);
			//System.out.println(i + ": " + ea.maxTimeStep());

			//System.out.println(i);
			//episodes.add(agent.runLearningEpisode(env));
			env.resetEnvironment();
		}
		
		double endTime = System.currentTimeMillis();
		System.out.println("Time: " + Double.toString(endTime-beginTime));
		simpleValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QFunction)agent), s, 
			hashingFactory, domain);

		// // Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
		// // new EpisodeSequenceVisualizer(v, domain, episodes);


		/* Run performance simulations - uncomment */

		/**
		 * Create factory for Q-learning agent
		 */
		// LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

		// 	@Override
		// 	public String getAgentName() {
		// 		return "Q-learning";
		// 	}

		// 	@Override
		// 	public LearningAgent generateAgent() {
		// 		return new rlQL(domain, discount, new SimpleHashableStateFactory(),
		// 			new ValueFunctionInitialization.ConstantValueFunctionInitialization(), 
		// 			learningRate, epsilon); //new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
		// 	}
		// };

		// //initial state generator
		// final ConstantStateGenerator sg = new ConstantStateGenerator(s);

		// //define learning environment
		// SimulatedEnvironment envNew = new SimulatedEnvironment(domain, rf, tf, sg);

		// //define experiment
		// LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(envNew,
		// 		1, 10000, qLearningFactory);

		// exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
		// 		PerformanceMetric.CUMULATIVESTEPSPEREPISODE, PerformanceMetric.AVERAGEEPISODEREWARD);


		// //start experiment
		// exp.startExperiment();






	}






	public static void main(String[] args) {
		List<Double> rewards = Arrays.asList(0.0);
		List<Double> penalties = Arrays.asList(-0.1);
		List<Double> transitions = Arrays.asList(0.8);
		List<Double> discounts = Arrays.asList(0.99);
		List<Double> learningRates = Arrays.asList(0.3);//, 0.7, 1.0);
		List<Double> epsilons = Arrays.asList(0.1);//, 0.1, 0.3);
		double precision = 0.001;
		int maxIterations = 2500;
		String outputPath = "output/";
		boolean vis = false;
		int count = 0;




		for(Double reward : rewards){
			for(Double penalty : penalties){
				for(Double transition : transitions){
					for(Double discount : discounts){
						for(Double learningRate : learningRates){
							for(Double epsilon : epsilons){
								System.out.format("4Room - QL: reward, penalty, transition, " +
									"discount: %f %f %f %f %f %f\n", reward, penalty, transition, 
									discount, learningRate, epsilon);
								qLearnGrid(false, 170, 3, 169, 1, transition, reward, 
									penalty, discount, learningRate, epsilon, precision, 
									maxIterations, outputPath, count, vis);
								count++;
							}
						}
					}
				}
			}
		}



























	}

}
