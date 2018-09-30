// Revision History.  This version contains the following features:
//
// Version 7 (April 10, 2018)
// This version was called autoParamCombo because it adds the feature of
// specifying vectors of one or more values for some parameters instead of
// being limited to a single value for each parameter.  The program then
// runs every possible combination of the set of values for each parameter.
// The parameters for which a vector of values can be supplied are listed
// in the function processAllParamCombos().  The elements of each vector
// are entered in this function.  Define a one element vector if the
// parameter is to take on only one value. Parameters for which this
// vector feature is not enabled are specified as #define constants early
// in the program. 
//
// Version 6 (March 27, 2018)
// This version was called graphTypeChoice because it provides the choice of
// using either the IGRAPH_WATTS_STROGATZ algorithm or the IGRAPH_ERDOS_RENYI
// algorithm to randomly generate the graph with the specified number of
// agents.
//
// Version 5 (March 12, 2018)
// This version was called socialProbAlgorithms because it uses a specified
// algorithm and parameter to compute the probability of using social
// communication for input at each tick (as opposed to prototype).
// It then makes a random choice between these alternatives with this
// probability.
//
// Version 4 (March 7, 2018)
// This version was called timing because it added the total runtime for all
// runs to the screen output.
//
// Version 3 (March 6, 2018)
// This version was called multirunFixed because it fixes a bug in
// multirun that caused a memory leak in Lens.  It may also add some detail.
// 
// Version 2 (January 2018)
// This version was called multirun because it adds the feature of running
// cMaango with identical parameters multiple times.  N_RUNS is the number of
// times to run.
//
// Version 1 (January 2018)
// This version was called multiagent and is an early version of cMaango.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <lens.h>
#include <util.h>
#include <network.h>
#include <igraph.h>

// Data Structures used by this model:

// (1) Prototypes
//	uber_prototype[i] (for each feature i)
//	prototype[a][i] for each agent a (obtained from distortion of uber_prototype)
//	exempler[i] for each epoch e and receiving agent a
//
// (2) Graph g (currently represented by igraph)
//	numberAgentConnections = number of edges in graph
//
// (3) Lens network for internals of each agent, used to obtain its output as a function of its input
//	lens("addNet agent_name ...");
//	weights developed by training, updated at each epoch
//
// (4) Outputs of each agent, updated at each epoch
#define MAX_AGENTS   500
#define MAX_FEATURES  50
// n_features was formerly N_INPUTS = N_OUTPUTS

#define N_RUNS     2

// To seed random number generator based on time, specify a negative number for SEED,
// otherwise SEED is used as the seed for all random number generators, including igraph.
#define SEED       12345

#define SOCIAL_PROB_ALGORITHM "constant"

#define LEARNING_RATE 0.05
#define MOMENTUM  0.9

// igraph parameters for IGRAPH_WATTS_STROGATZ
#define NEIGHBORHOOD 4
#define PROB_REWIRE  0.10

#define CMDLEN    100000

#define DISPLAY_TO_SCREEN 0
#define SAVE_WEIGHTS 0
#define STORE_AGENT_CONNECTIONS 1
#define OMIT_ROWS_FOR_AGENTS_NOT_UPDATED 1

// defining CONNECTION_TYPE to be an IGRAPH._.. causes agent connection to be represented by edges in igraph network 
#define USE_IGRAPH

int n_agents;
int n_ticks;
int n_features;
real proportion_hidden;
int n_hidden;
real proto_p_on;
real proto_p_flip;
real item_p_flip;
real social_prob_parameter;


real outputs[MAX_AGENTS][MAX_FEATURES]; // current outputs of each agent
			           // agents are numbered from 0 to n_agents-1 for compatibility with igraph
real uber_prototype[MAX_FEATURES];
real prototype[MAX_AGENTS][MAX_FEATURES];
int numberAgentConnections;

int rand_int(int max);  // random int from 0 to max-1

// The representation of the graph is contained withion this section, including the #else.

#ifdef USE_IGRAPH 
typedef enum {IGRAPH_WATTS_STROGATZ, IGRAPH_ERDOS_RENYI} Igraph_type;

igraph_t graph;
Igraph_type graphType = IGRAPH_WATTS_STROGATZ;

void initAgentConnections(int runNum)
{
    // The actions depending on the type of igraph are contained within this switch statement.
    // This switch statement sets up the type of graph selected above.
    switch(graphType)
    {
        case IGRAPH_WATTS_STROGATZ:
        {
            int error1 = igraph_watts_strogatz_game(&graph, 1, n_agents, NEIGHBORHOOD, PROB_REWIRE, 0, 0);
            int error2 = igraph_to_directed(&graph, IGRAPH_TO_DIRECTED_MUTUAL); // creates bi-directional graph

            if (error1 || error2)
	        exit(1); 

	    break;
        }

        case IGRAPH_ERDOS_RENYI:
        {
            igraph_erdos_renyi_t graphSubtype = IGRAPH_ERDOS_RENYI_GNM;
            igraph_real_t p_or_m = round(0.05 * n_agents * n_agents);

            int error1 = igraph_erdos_renyi_game(&graph, graphSubtype, n_agents, p_or_m, 0, 0);
            int error2 = igraph_to_directed(&graph, IGRAPH_TO_DIRECTED_MUTUAL); // creates bi-directional graph

            if (error1 || error2)
	        exit(1); 

	    break;
        }

	default:
	    printf("INVALID IGRAPH TYPE");
	    exit(1);
    }

    // The following code is executed for any type of igraph:

    numberAgentConnections = (int)igraph_ecount(&graph);
    printf("there are %d agent connections\n\n", numberAgentConnections);

    if (STORE_AGENT_CONNECTIONS)
    {
	FILE *fp;
	char filename[40];

	sprintf(filename, "connections_%d.txt", runNum);
	fp = fopen(filename, "w");

	for (igraph_integer_t ac = 0; (int)ac < numberAgentConnections; ac++) // ac = each edge id
	{
	    igraph_integer_t sender, receiver;
	    igraph_edge(&graph, ac, &sender, &receiver);
	    fprintf(fp, "%d %d\n", (int)receiver, (int)sender);
	}

	fclose(fp);
    }
}

void chooseRandomConnection(int *pReceiver, int *pSender)
{
    igraph_integer_t sender, receiver;
    igraph_integer_t ac = rand_int(numberAgentConnections); // random# from 0 thru numberAgentConnections - 1
//    igraph_integer_t ac = rand() % numberAgentConnections; // random# from 0 thru numberAgentConnections - 1
    igraph_edge(&graph, ac, &sender, &receiver);

    *pReceiver = (int)receiver;
    *pSender   = (int)sender;
}

#else

typedef struct AgentLink
{
	int receiver;
	int sender;
} AgentLink;

AgentLink agentConnections[] = { {0,2}, {1,0}, {2,1} };

void initAgentConnections(int runNum)
{
    numberAgentConnections = sizeof(agentConnections) / sizeof(agentConnections[0]);
    printf("there are %d agent connections\n\n", numberAgentConnections);

    if (STORE_AGENT_CONNECTIONS)
    {
	FILE *fp;
	char filename[40];

	sprintf(filename, "connections_%d.txt", runNum);
	fp = fopen(filename, "w");

	for (int ac = 0; ac < numberAgentConnections; ac++)
	{
	    AgentLink connection = agentConnections[ac];
	    fprintf(fp, "%d %d\n", connection.receiver, connection.sender);
	}

	fclose(fp);
    }
}

void chooseRandomConnection(int *pReceiver, int *pSender)
{
    int ac = rand_int(numberAgentConnections); // random# from 0 thru numberAgentConnections - 1
//  int ac = rand() % numberAgentConnections; // random# from 0 thru numberAgentConnections - 1

    *pReceiver = agentConnections[ac].receiver;
    *pSender   = agentConnections[ac].sender;
}

#endif

real rand_real() // uniform;ly dist 0.0 to 1.0
{
    return drand48(); // drand48 is supposed to produce a better random number than rand()
//	return (real)rand() / (real)RAND_MAX;
}

real rand_real_a_to_b(real a, real b) // uniformly dist real between a and b
{
	return (b - a) * rand_real() + a;
}

int rand_int(int max)  // random int from 0 to max-1
{
    return floor((real)max * drand48());
}


int oneWithProb(real p)
{
	return (rand_real() < p) ? 1 : 0;
}


void storeParameters(int runNum)
{
    FILE *fp;
    char filename[40];

    sprintf(filename, "parameters_%d.txt", runNum);
    fp = fopen(filename, "w");

    fprintf(fp, "n_agents %d\n",   n_agents);
    fprintf(fp, "n_ticks %d\n",    n_ticks);
    fprintf(fp, "n_features %d\n", n_features);
    fprintf(fp, "proportion_hidden %f\n", proportion_hidden);
    fprintf(fp, "n_hidden %d\n",   n_hidden);
    fprintf(fp, "N_RUNS %d\n",     N_RUNS);

    
    if (SEED < 0)
        fprintf(fp, "SEED based on time\n");
    else
        fprintf(fp, "SEED %d\n", SEED);

    fprintf(fp, "proto_p_on %f\n",   proto_p_on);
    fprintf(fp, "proto_p_flip %f\n", proto_p_flip);
    fprintf(fp, "item_p_flip %f\n",  item_p_flip);

    fprintf(fp, "SOCIAL_PROB_ALGORITHM %s\n", SOCIAL_PROB_ALGORITHM);
    fprintf(fp, "social_prob_parameter %f\n", social_prob_parameter);

    fprintf(fp, "LEARNING_RATE %f\n", LEARNING_RATE);
    fprintf(fp, "MOMENTUM %f\n",      MOMENTUM);

#ifdef USE_IGRAPH 
    if (graphType == IGRAPH_WATTS_STROGATZ)
    {
        fprintf(fp, "GRAPH_TYPE IGRAPH_WATTS_STROGATZ\n");
        fprintf(fp, "NEIGHBORHOOD %d\n", NEIGHBORHOOD);
        fprintf(fp, "PROB_REWIRE %f\n",  PROB_REWIRE);
    }
#endif

    fclose(fp);

    printf("n_agents %d\n",   n_agents);
    printf("n_ticks %d\n",    n_ticks);
    printf("n_features %d\n", n_features);
    printf("proportion_hidden %f\n", proportion_hidden);
    printf("n_hidden %d\n",   n_hidden);
    printf("proto_p_on %f\n",   proto_p_on);
    printf("proto_p_flip %f\n", proto_p_flip);
    printf("item_p_flip %f\n",  item_p_flip);
    printf("social_prob_parameter %f\n", social_prob_parameter);
    printf("runNum %d\n\n\n\n", runNum);
}

// save inputs & outputs of agent
void saveInputsOutputs(real *inputs, real *outputs)	// from tlnes.c
{
  int i ;
  int nIn = Net->numInputs ;
  int nOut = Net->numOutputs ;
  for (i = 0 ; i < nIn  ; i++) inputs[i] = Net->input[i]->output ;
  for (i = 0 ; i < nOut ; i++) outputs[i] = Net->output[i]->output ;
}

// save outputs of agent
void saveOutputs(real *outs)	// from tlens.c
{
  int i ;
  int nOut = Net->numOutputs ;
  for (i = 0 ; i < nOut ; i++) outs[i] = Net->output[i]->output ;
}

void printVector(real* vec, int len)
{
  int i ;

  if (!DISPLAY_TO_SCREEN) return;

  for (i = 0 ; i < len  ; i++) printf("%.2f ", vec[i]) ;
  printf("\n");
}

void fprintVector(FILE *fp, real* vec, int len)
{
  int i ;

  for (i = 0 ; i < len  ; i++) fprintf(fp, "%.2f ", vec[i]) ;
  fprintf(fp, "\n");
}


void createExampleSet(real *inputs, real *targets)
{
  // uses fixed example set name: train
  // assumes all nets have same sized inputs/outputs
  // this copies inputs to inputs and targets to targets
  int i, pos ;
  int nIn = Net->numInputs ;
  int nOut = Net->numOutputs ;
  char cmd[CMDLEN];
//  printf("in=%d, out=%d\n", nIn, nOut);

  pos = sprintf(cmd, "loadExamples \"|echo \\\"I: ");
  for (i = 0 ; i < nIn  ; i++)
  {
     pos += sprintf(cmd+pos, "%.0f ", inputs[i]);
//     printf("******** length(cmd)=%ld, pos=%d **********\n\n", strlen(cmd), pos);
//     printf("cmd=%s\n\n", cmd);
  }

  pos += sprintf(cmd+pos, " T: ");
  for (i = 0 ; i < nOut  ; i++)
  {
      pos += sprintf(cmd+pos, "%.0f ", targets[i]);
//     printf("********* length(cmd)=%ld, pos=%d **********\n\n", strlen(cmd), pos);
//     printf("cmd=%s\n\n", cmd);
  }

  sprintf(cmd+pos, ";\\\"\" -s train -mode REPLACE\n");
//  printf("ready to call createExampleSet/lens(cmd)\n");
  lens(cmd);
}

// dcp
void overwriteExample(real *inputs, real *targets)
{
  // will overwrite first event of first example of current training set
  int i ;
  int nIn = Net->numInputs ;
  int nOut = Net->numOutputs ;
  real *exInputs  = Net->trainingSet->firstExample->event->input->val ;
  real *exTargets = Net->trainingSet->firstExample->event->target->val ;
  for (i = 0 ; i < nIn  ; i++) exInputs[i] = inputs[i] ;
  for (i = 0 ; i < nOut ; i++) exTargets[i] = targets[i] ;
}


     // computes outputs for agent from its inputs
     // leaves result in array referenced by outs.
void computeOutputs(int agent, real *ins, real *outs, int tick)
{
    lens("useNet agent%d", agent);

    if (SAVE_WEIGHTS && (tick == 0))
        lens("saveWeights weights_tick_%d_agent_%d.wt", tick, agent);

//    printf("about to overwrite example for agent %d\n", agent);
    overwriteExample(ins, ins);
//    printf("about to train 1\n");
    lens("train 1");
//    printf("train 1 completed\n");
    saveOutputs(outs);
}


void printOutputs(int i)	// print outputs from agent i
{
        if (!DISPLAY_TO_SCREEN) return;

	printf("outputs from agent %d:  ", i);

	for (int outNum = 0; outNum < n_features; outNum++)
	{
		printf("%f  ", outputs[i][outNum]);
	}

	printf("\n");
}

void printAllOutputs()	// print outputs from all agents
{
        if (!DISPLAY_TO_SCREEN) return;

	for (int i = 0; i < n_agents; i++)
	{
		printOutputs(i);
	}
}


void initializeRun(int runNum)
{
    int a, i;
    FILE *fp;
    char filename[40];

    if (!DISPLAY_TO_SCREEN)
        lens("verbosity 0");

    sprintf(filename, "prototypes_%d.txt", runNum);
    fp = fopen(filename, "w");

    // initialize prototypes

    // create uber (main) prototype (each feature on with prob proto_p_on)

    for (i = 0 ; i < n_features ; i++)
        uber_prototype[i] = oneWithProb(proto_p_on);

    if (DISPLAY_TO_SCREEN) printf("Uber:    ") ;
    printVector(uber_prototype, n_features);

    fprintf(fp, "U ");
    fprintVector(fp, uber_prototype, n_features);

    // create agent-specific prototypes as distortions of uber prototype
    // (with prob proto_p_flip, regenerate feature (1 with prob proto_p_on))
    for (a = 0 ; a < n_agents ; a++)
    {
        for (i = 0 ; i < n_features ; i++)
            prototype[a][i] = (oneWithProb(proto_p_flip) ? oneWithProb(proto_p_on) : uber_prototype[i]); 

        if (DISPLAY_TO_SCREEN) printf("Proto %d: ", a);
        printVector(prototype[a], n_features) ;

        fprintf(fp, "%d ", a);
        fprintVector(fp, prototype[a], n_features);

    }

    fclose(fp);

    initAgentConnections(runNum); // initiialize graph of agent_networonetwork

    for (a = 0 ; a < n_agents ; a++)
    {
        lens("addNet agent%d %d %d %d", a, n_features, n_hidden, n_features);
        lens("setObj learningRate %f", LEARNING_RATE);
        lens("setObj momentum %f", MOMENTUM);
	lens("setObj batchSize 1"); // DO WE NEED THIS?
	lens("setObj reportInterval 1");  // DO WE NEED THIS?
        lens("resetNet");
    }

    storeParameters(runNum);
}


void concludeRun(int runNum)
{
    lens("deleteNets *"); // delete all networks because they will be recreated on next run
    printf("\nrunNum %d completed\n\n", runNum);

#ifdef USE_IGRAPH 
    igraph_destroy(&graph);
#endif
}


// sets distortedProto to a random distortion of proto
void distortAgentPrototype(real *proto, real *distortedProto)
{
    int i;

    for (i = 0 ; i < n_features ; i++)
        distortedProto[i] = oneWithProb(item_p_flip) ? oneWithProb(proto_p_on) : proto[i];
}


     // load a network for each agent ans separately pretrain it
void pretraining(FILE *fp)
{
    int a;  // agent#
    int i;  // features
    real inputs[n_features];

    // load and pretrain each agent
    if (DISPLAY_TO_SCREEN) printf("outputs of PRETRAINING (one epoch):\n");

    for (a = 0 ; a < n_agents ; a++)
    {
        // distort agent-specific prototype to create input for epoch 0
        distortAgentPrototype(prototype[a], inputs);

        lens("useNet agent%d", a);
        if (a == 0)
        {
//             printf("about to create example set\n");
             createExampleSet(inputs, inputs);
//             printf("created example set\n");
        }
        lens("useTrainingSet train") ;


        computeOutputs(a, inputs, outputs[a], 0); // loads inputs (as both inputs and targets) 
        // saves outputs in outputs[a] since it will be the initial output value for
	// iterations in processRun.

        if (DISPLAY_TO_SCREEN) printf("Agent %d: ", a) ;
        printVector(outputs[a], n_features);

	// For tick 0, store initial data for each agent in format:
	// <tick#> <agent#> - - <inputs> <outputs>
        fprintf(fp, "0 %d - - ", a);  // tick 0 and agent number

        for (i = 0;  i < n_features; i++)
            fprintf(fp, "%f ", inputs[i]);

        for (i = 0;  i < n_features; i++)
            fprintf(fp, "%f ", outputs[a][i]);

        fprintf(fp, "\n");
    }
}


int usingSocialForInput(int tick)
{
    real p;  // probability of using social communication for input at tick (as opposed to prototype)

    real x = (real)tick / (real)n_ticks;

    if ( strcmp(SOCIAL_PROB_ALGORITHM, "constant") == 0 )
    {
        p = social_prob_parameter;  // in this case, the constant is the parameter
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "linear") == 0 )
    {
        p = x;
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "logistic_increasing") == 0 )
    {
        x -= 0.5;
        p =  1.0 / (1.0 + exp(-social_prob_parameter * x)); // prob of using social communication
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "logistic_decreasing") == 0 )
    {
        x -= 0.5;
        p =  1.0 - 1.0 / (1.0 + exp(-social_prob_parameter * x)); // prob of using social communication
    }
    else
    {
        printf("INVALID SOCIAL_PROB_ALGORITHM");
        exit(1);
    }

//    printf("prob of using output of sender (social communication) for input at tick %d = %f\n", tick, p);
//    printf("computed from %s algorithm using parameter %f\n", SOCIAL_PROB_ALGORITHM, social_prob_parameter);

    if (rand_real() < p)
	return 1;  // use output of sender (social communication)
    else
        return 0;  // use distorted prototype
}


void processRun(int runNum)
{
    int tick;	// tick #
    FILE *fp;
    char filename[40];

    sprintf(filename, "history_%d.txt", runNum);
    fp = fopen(filename, "w");

    fprintf(fp, "<tick#> <agent#> <1 if receiving agent> <sending agent#> <%d inputs> <%d outputs>\n\n", n_features, n_features);

    initializeRun(runNum);
    pretraining(fp);
    printAllOutputs();	// starting outputs

  // for some number of iterations, select FROM and TO randomly, then
  // train TO on last output of FROM (saved in outputs[FROM])

        if (DISPLAY_TO_SCREEN) printf("\nCOMMUNICATION or distorted prototype:\n");

	for (tick = 1; tick <= n_ticks; tick++)
	{
		int receiver, sender;  // agent # of receiving agent and sending agent
                int useProto;
		real inputsReceiver[n_features];

		chooseRandomConnection(&receiver, &sender);
		// agent receiver's input gets agent sender's output
		// we assume here that the number of inputs and outputs are both = n_features.  Otherwise a transformation
		// function would have to be applied to the output.

                if (DISPLAY_TO_SCREEN) printf("\nat tick %d:\n", tick);

                if ((useProto = !usingSocialForInput(tick)))
	        {
                    if (DISPLAY_TO_SCREEN) printf("agent %d uses distortion of its prototype for input\n", receiver);

                    // distort agent-specific prototype to create input for receiver
                    distortAgentPrototype(prototype[receiver], inputsReceiver);
                }
                else
                {
                    if (DISPLAY_TO_SCREEN) printf("agent %d receives output of agent %d\n", receiver, sender);

	    	    for (int i = 0;  i < n_features; i++)
		        inputsReceiver[i] = outputs[sender][i];
                }

                if (DISPLAY_TO_SCREEN) printf("useProto = %d\n", useProto);

		computeOutputs(receiver, inputsReceiver, outputs[receiver], tick);
		printAllOutputs();

		// For tick, store initial data for each agent in format:
		// <tick#> <agent#> <1 if receiving agent>  <sending agent#> <inputs> <outputs>

		for (int a = 0; a < n_agents; a++) // agent number
		{
                    if (OMIT_ROWS_FOR_AGENTS_NOT_UPDATED && (a != receiver))
                        continue;

		    fprintf(fp, "%d %d ", tick, a);

		    if (a == receiver)
		    {
                        if (useProto)
                            fprintf(fp, "1 P ");
                        else
			    fprintf(fp, "1 %d ", sender);
		    }
		    else
		    {
			fprintf(fp, "0 - ");
		    }

		    if (a == receiver)
		    {
	    		for (int i = 0;  i < n_features; i++)
	        	    fprintf(fp, "%f ", inputsReceiver[i]);
		    }
		    else
		    {
	    		for (int i = 0;  i < n_features; i++)
			    fprintf(fp, "- ");
		    }

	    	    for (int i = 0;  i < n_features; i++)
	        	fprintf(fp, "%f ", outputs[a][i]);

	            fprintf(fp, "\n");
		}

	}

	concludeRun(runNum);
	fclose(fp);
}

void processAllParamCombos(void)
{
    int runNum;
    char dirstr[1000] = ""; // name of subdirectory for output data of this parameter combination
    char cmd[1024] = "";

    int v_n_agents[] = { 100, 1000, 2000 }; // vector of values for n_agents
    int l_n_agents = sizeof(v_n_agents)/sizeof(v_n_agents[0]); // length of v_n_agents
    int i_n_agents;             // index into v_n_agents to identify value being used

    int v_n_features[] = { 20, 40 };
    int l_n_features = sizeof(v_n_features)/sizeof(v_n_features[0]);
    int i_n_features;  // index into v_n_features

    real v_proportion_hidden[] = { 0.3, 0.5 };
    int  l_proportion_hidden = sizeof(v_proportion_hidden)/sizeof(v_proportion_hidden[0]);
    int  i_proportion_hidden;

    real v_proto_p_on[] = { 0.5 };
    int  l_proto_p_on = sizeof(v_proto_p_on)/sizeof(v_proto_p_on[0]);
    int  i_proto_p_on;

    real v_proto_p_flip[] = { 0.2 };
    int  l_proto_p_flip = sizeof(v_proto_p_flip)/sizeof(v_proto_p_flip[0]);
    int  i_proto_p_flip;

    real v_item_p_flip[] = { 0.1 };
    int  l_item_p_flip = sizeof(v_item_p_flip)/sizeof(v_item_p_flip[0]);
    int  i_item_p_flip;

    real v_social_prob_parameter[] = { 0.2 };
    int  l_social_prob_parameter = sizeof(v_social_prob_parameter)/sizeof(v_social_prob_parameter[0]);
    int  i_social_prob_parameter;


    for (i_n_agents = 0; i_n_agents < l_n_agents; i_n_agents++)
    {
      n_agents = v_n_agents[i_n_agents];
      n_ticks  = 100 * n_agents;

      for (i_n_features = 0; i_n_features < l_n_features; i_n_features++)
      {
        n_features = v_n_features[i_n_features];

        for (i_proportion_hidden = 0; i_proportion_hidden < l_proportion_hidden; i_proportion_hidden++)
        {
          proportion_hidden = v_proportion_hidden[i_proportion_hidden];
          n_hidden = (int)round(proportion_hidden * (real)n_features);

          for (i_proto_p_on = 0; i_proto_p_on < l_proto_p_on; i_proto_p_on++)
          {
            proto_p_on = v_proto_p_on[i_proto_p_on];

            for (i_proto_p_flip = 0; i_proto_p_flip < l_proto_p_flip; i_proto_p_flip++)
            {
              proto_p_flip = v_proto_p_flip[i_proto_p_flip];

              for (i_item_p_flip = 0; i_item_p_flip < l_item_p_flip; i_item_p_flip++)
              {
                item_p_flip = v_item_p_flip[i_item_p_flip];

                for (i_social_prob_parameter = 0; i_social_prob_parameter < l_social_prob_parameter; i_social_prob_parameter++)
                {
                  social_prob_parameter = v_social_prob_parameter[i_social_prob_parameter];

                  strcpy(dirstr, "");

                  if (l_n_agents > 1)
                    sprintf(dirstr+strlen(dirstr), "agents%d_", n_agents);

                  if (l_n_features > 1)
                    sprintf(dirstr+strlen(dirstr), "features%d_", n_features);

                  if (l_proportion_hidden > 1)
                    sprintf(dirstr+strlen(dirstr), "phidden%.3f_", proportion_hidden);

                  if (l_proto_p_on > 1)
                    sprintf(dirstr+strlen(dirstr), "protopOn%.3f_", proto_p_on);

                  if (l_proto_p_flip > 1)
                    sprintf(dirstr+strlen(dirstr), "protopFlip%.3f_", proto_p_flip);

                  if (l_item_p_flip > 1)
                    sprintf(dirstr+strlen(dirstr), "itempFlip%.3f_", item_p_flip);

                  if (l_social_prob_parameter > 1)
                    sprintf(dirstr+strlen(dirstr), "socialpParam%.3f_", social_prob_parameter);

                  if (strlen(dirstr) > 0)  // more than one parameter combination is being run
                  {
                    dirstr[strlen(dirstr)-1] = '\0'; // remove the final underscore
                    strcpy(cmd, "mkdir ");
                    strcat(cmd, dirstr);
                    system(cmd);  // create subdirectory for data of this parameter combination
//                    strcpy(cmd, "cd ");
//                    strcat(cmd, dirstr);
//                    system(cmd);  // send file output to this subdirectory
                    int res = chdir(dirstr);
                    printf("chdir returns %d\n", res);
                    printf("For data output, getcwd=%s\n\n", getcwd(NULL,0));
                  }

    if (SEED < 0)
        srand48(time(NULL));   // seed random number generator
    else
    {
        srand48(SEED);

#ifdef USE_IGRAPH 
        igraph_rng_seed(igraph_rng_default(), SEED); // igraph uses a separate random number generator
#endif
    }

                  for (runNum = 0; runNum < N_RUNS; runNum++)
                  {
                    processRun(runNum);  // includes addNet and resetNet before the run and deleteNets * afterward.
                  }

                  if (strlen(dirstr) > 0)
                  {
                    int res = chdir("..");  // return to program directory
                    printf("chdir returns %d\n", res);
                    printf("Back to program directory: getcwd=%s\n\n", getcwd(NULL,0));
                  }

                }
              }
            }
          }
        }
      }
    }
}

int main(int argc, char* argv[])
{
    clock_t timer;

    timer = clock();

//  srand((unsigned) time(NULL)); // seed random number generator
    system("rm *.txt"); // remove output files from previous runs in this directory

    if (startLens(argv[0], 1))
    {
        fprintf(stderr, "Lens Failed\n");
        exit(1);
    }

    processAllParamCombos();

    timer = clock() - timer;
    double seconds = ((double)timer)/CLOCKS_PER_SEC;
    printf("program took %.3f seconds\n", seconds);    
}

