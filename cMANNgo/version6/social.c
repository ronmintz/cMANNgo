// Revision History.  This version contains the following features:
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
#define N_AGENTS     100
#define N_FEATURES   20
// N_FEATURES was formerly N_INPUTS = N_OUTPUTS

#define PROPORTION_HIDDEN 0.3
#define N_HIDDEN ((int)round((N_FEATURES) * (PROPORTION_HIDDEN)))

#define N_TICKS    10000
#define N_RUNS     2

// To seed random number generator based on time, specify a negative number for SEED,
// otherwise SEED is used as the seed for all random number generators, including igraph.
#define SEED       12345

#define PROTO_P_ON   0.5
#define PROTO_P_FLIP 0.2
#define ITEM_P_FLIP  0.1
#define SOCIAL_PROB_ALGORITHM "constant"
#define SOCIAL_PROB_PARAMETER 0.2

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

real outputs[N_AGENTS][N_FEATURES]; // current outputs of each agent
			           // agents are numbered from`0 to N_AGENTS-1 for compatibility with igraph
real uber_prototype[N_FEATURES];
real prototype[N_AGENTS][N_FEATURES];
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
            int error1 = igraph_watts_strogatz_game(&graph, 1, N_AGENTS, NEIGHBORHOOD, PROB_REWIRE, 0, 0);
            int error2 = igraph_to_directed(&graph, IGRAPH_TO_DIRECTED_MUTUAL); // creates bi-directional graph

            if (error1 || error2)
	        exit(1); 

	    break;
        }

        case IGRAPH_ERDOS_RENYI:
        {
            igraph_erdos_renyi_t graphSubtype = IGRAPH_ERDOS_RENYI_GNM;
            igraph_real_t p_or_m = round(0.05 * N_AGENTS * N_AGENTS);

            int error1 = igraph_erdos_renyi_game(&graph, graphSubtype, N_AGENTS, p_or_m, 0, 0);
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

    fprintf(fp, "N_AGENTS %d\n",   N_AGENTS);
    fprintf(fp, "N_FEATURES %d\n", N_FEATURES);
    fprintf(fp, "PROPORTION_HIDDEN %f\n", PROPORTION_HIDDEN);
    fprintf(fp, "N_HIDDEN %d\n",   N_HIDDEN);
    fprintf(fp, "N_TICKS %d\n",    N_TICKS);
    fprintf(fp, "N_RUNS %d\n",     N_RUNS);

    
    if (SEED < 0)
        fprintf(fp, "SEED based on time\n");
    else
        fprintf(fp, "SEED %d\n", SEED);

    fprintf(fp, "PROTO_P_ON %f\n",   PROTO_P_ON);
    fprintf(fp, "PROTO_P_FLIP %f\n", PROTO_P_FLIP);
    fprintf(fp, "ITEM_P_FLIP %f\n",  ITEM_P_FLIP);

    fprintf(fp, "SOCIAL_PROB_ALGORITHM %s\n", SOCIAL_PROB_ALGORITHM);
    fprintf(fp, "SOCIAL_PROB_PARAMETER %f\n", SOCIAL_PROB_PARAMETER);

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

	for (int outNum = 0; outNum < N_FEATURES; outNum++)
	{
		printf("%f  ", outputs[i][outNum]);
	}

	printf("\n");
}

void printAllOutputs()	// print outputs from all agents
{
        if (!DISPLAY_TO_SCREEN) return;

	for (int i = 0; i < N_AGENTS; i++)
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

    // create uber (main) prototype (each feature on with prob PROTO_P_ON)

    for (i = 0 ; i < N_FEATURES ; i++)
        uber_prototype[i] = oneWithProb(PROTO_P_ON);

    if (DISPLAY_TO_SCREEN) printf("Uber:    ") ;
    printVector(uber_prototype, N_FEATURES);

    fprintf(fp, "U ");
    fprintVector(fp, uber_prototype, N_FEATURES);

    // create agent-specific prototypes as distortions of uber prototype
    // (with prob PROTO_P_FLIP, regenerate feature (on with prob PROTO_P_ON))
    for (a = 0 ; a < N_AGENTS ; a++)
    {
        for (i = 0 ; i < N_FEATURES ; i++)
            prototype[a][i] = (oneWithProb(PROTO_P_FLIP) ? oneWithProb(PROTO_P_ON) : uber_prototype[i]); 

        if (DISPLAY_TO_SCREEN) printf("Proto %d: ", a);
        printVector(prototype[a], N_FEATURES) ;

        fprintf(fp, "%d ", a);
        fprintVector(fp, prototype[a], N_FEATURES);

    }

    fclose(fp);

    initAgentConnections(runNum); // initiialize graph of agent network

    for (a = 0 ; a < N_AGENTS ; a++)
    {
        lens("addNet agent%d %d %d %d", a, N_FEATURES, N_HIDDEN, N_FEATURES);
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

    for (i = 0 ; i < N_FEATURES ; i++)
        distortedProto[i] = oneWithProb(ITEM_P_FLIP) ? oneWithProb(PROTO_P_ON) : proto[i];
}


     // load a network for each agent and separately pretrain it
void pretraining(FILE *fp)
{
    int a;  // agent#
    int i;  // features
    real inputs[N_FEATURES];

    // load and pretrain each agent
    if (DISPLAY_TO_SCREEN) printf("outputs of PRETRAINING (one epoch):\n");

    for (a = 0 ; a < N_AGENTS ; a++)
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
        printVector(outputs[a], N_FEATURES);

	// For tick 0, store initial data for each agent in format:
	// <tick#> <agent#> - - <inputs> <outputs>
        fprintf(fp, "0 %d - - ", a);  // tick 0 and agent number

        for (i = 0;  i < N_FEATURES; i++)
            fprintf(fp, "%f ", inputs[i]);

        for (i = 0;  i < N_FEATURES; i++)
            fprintf(fp, "%f ", outputs[a][i]);

        fprintf(fp, "\n");
    }
}


int usingSocialForInput(int tick)
{
    real p;  // probability of using social communication for input at tick (as opposed to prototype)

    real x = (real)tick / (real)N_TICKS;

    if ( strcmp(SOCIAL_PROB_ALGORITHM, "constant") == 0 )
    {
        p = SOCIAL_PROB_PARAMETER;  // in this case, the constant is the parameter
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "linear") == 0 )
    {
        p = x;
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "logistic_increasing") == 0 )
    {
        x -= 0.5;
        p =  1.0 / (1.0 + exp(-SOCIAL_PROB_PARAMETER * x)); // prob of using social communication
    }
    else if ( strcmp(SOCIAL_PROB_ALGORITHM, "logistic_decreasing") == 0 )
    {
        x -= 0.5;
        p =  1.0 - 1.0 / (1.0 + exp(-SOCIAL_PROB_PARAMETER * x)); // prob of using social communication
    }
    else
    {
        printf("INVALID SOCIAL_PROB_ALGORITHM");
        exit(1);
    }

//    printf("prob of using output of sender (social communication) for input at tick %d = %f\n", tick, p);
//    printf("computed from %s algorithm using parameter %f\n", SOCIAL_PROB_ALGORITHM, SOCIAL_PROB_PARAMETER);

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

    fprintf(fp, "<tick#> <agent#> <1 if receiving agent> <sending agent#> <%d inputs> <%d outputs>\n\n", N_FEATURES, N_FEATURES);

    initializeRun(runNum);
    pretraining(fp);
    printAllOutputs();	// starting outputs

  // for some number of iterations, select FROM and TO randomly, then
  // train TO on last output of FROM (saved in outputs[FROM])

        if (DISPLAY_TO_SCREEN) printf("\nCOMMUNICATION or distorted prototype:\n");

	for (tick = 1; tick <= N_TICKS; tick++)
	{
		int receiver, sender;  // agent # of receiving agent and sending agent
                int useProto;
		real inputsReceiver[N_FEATURES];

		chooseRandomConnection(&receiver, &sender);
		// agent receiver's input gets agent sender's output
		// we assume here that the number of inputs and outputs are both = N_FEATURES.  Otherwise a transformation
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

	    	    for (int i = 0;  i < N_FEATURES; i++)
		        inputsReceiver[i] = outputs[sender][i];
                }

                if (DISPLAY_TO_SCREEN) printf("useProto = %d\n", useProto);

		computeOutputs(receiver, inputsReceiver, outputs[receiver], tick);
		printAllOutputs();

		// For tick, store initial data for each agent in format:
		// <tick#> <agent#> <1 if receiving agent>  <sending agent#> <inputs> <outputs>

		for (int a = 0; a < N_AGENTS; a++) // agent number
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
	    		for (int i = 0;  i < N_FEATURES; i++)
	        	    fprintf(fp, "%f ", inputsReceiver[i]);
		    }
		    else
		    {
	    		for (int i = 0;  i < N_FEATURES; i++)
			    fprintf(fp, "- ");
		    }

	    	    for (int i = 0;  i < N_FEATURES; i++)
	        	fprintf(fp, "%f ", outputs[a][i]);

	            fprintf(fp, "\n");
		}

	}

	concludeRun(runNum);
	fclose(fp);
}

int main(int argc, char* argv[])
{
    int runNum;
    clock_t timer;

    timer = clock();

//  srand((unsigned) time(NULL)); // seed random number generator
    system("rm *.txt"); // remove output files from previous runs in this directory

    if (startLens(argv[0], 1))
    {
        fprintf(stderr, "Lens Failed\n");
        exit(1);
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

    timer = clock() - timer;
    double seconds = ((double)timer)/CLOCKS_PER_SEC;
    printf("program took %.3f seconds\n", seconds);    
}

