// Revision History:
// Version 1 (January 2018)
// This version was called multiagent and is an early version of cMaango.

#include <stdio.h>
#include <stdlib.h>
#include <lens.h>
#include <util.h>
#include <network.h>
#include <time.h>

#define N_RUNS 10
#define N_AGENTS    1000
#define N_FEATURES  40
#define N_HIDDEN    40

#define PRETRAIN_EPOCHS 20

#define COMMUNICATION_ITERATIONS 100000

#define PROTO_P_ON   0.2
#define PROTO_P_FLIP 0.4
#define ITEM_P_FLIP  0.2

#define CMDLEN    1024

#define WITH_PROB(p) (drand48() < (p) ? 1 : 0)

// 0 to max-1
#define RAND_INT(max) (floor((max)*drand48()))

void saveOutputs(real *outputs) {
  int i ;
  int nOut = Net->numOutputs ;
  for (i = 0 ; i < nOut ; i++) outputs[i] = Net->output[i]->output ;
}

void printVector(real* vec, int len) {
  int i ;
  return;

  for (i = 0 ; i < len  ; i++) printf("%.2f ", vec[i]) ;
  printf("\n");
}

void printOutputs() {
  int i ;
  int nOut = Net->numOutputs ;
  return;

  for (i = 0 ; i < nOut  ; i++) printf("%f ", Net->output[i]->output) ;
  printf("\n");
}

void createLoadExample(real *inputs, real *outputs) {
  // uses fixed example set name: train
  // assumes all nets have same sized inputs/outputs
  // this copies inputs to inputs and outputs to targets
  int i, pos ;
  int nIn = Net->numInputs ;
  int nOut = Net->numOutputs ;
  char cmd[CMDLEN];
  pos = sprintf(cmd, "loadExamples \"|echo \\\"I: ");
  for (i = 0 ; i < nIn  ; i++) pos += sprintf(cmd+pos, "%f ", inputs[i]);
  pos += sprintf(cmd+pos, " T: ");
  for (i = 0 ; i < nOut  ; i++) pos += sprintf(cmd+pos, "%f ", outputs[i]);
  sprintf(cmd+pos, ";\\\"\" -s train -mode REPLACE\n");
  lens(cmd);
}

int main(int argc, char *argv[]) {
  int a, i, e ;
  real outputs[N_AGENTS][N_FEATURES] ;
  real prototype[N_AGENTS][N_FEATURES] ;
  real uber_prototype[N_FEATURES] ;
  real exemplar[N_FEATURES] ;
  int runNum;

  // seed RNG
  srand48(time(NULL)) ;

  // start lens
  if (startLens(argv[0], 1)) {
    fprintf(stderr, "Lens Failed\n");
    exit(1);
  }

  lens("verbosity 0");

for (runNum = 0; runNum < N_RUNS; runNum++)
{
  // create uber (main) prototype (each feature on with prob PROTO_P_ON)
  for (i = 0 ; i < N_FEATURES ; i++)
    uber_prototype[i] = WITH_PROB(PROTO_P_ON) ;
//  printf("Uber:    ") ;
  printVector(uber_prototype, N_FEATURES) ;

  // create agent-specific prototypes as distortions of uber prototype
  // (with prob PROTO_P_FLIP, regenerate feature (on with prob PROTO_P_ON))
  for (a = 0 ; a < N_AGENTS ; a++) {
    for (i = 0 ; i < N_FEATURES ; i++)
      prototype[a][i] = (WITH_PROB(PROTO_P_FLIP) ? WITH_PROB(PROTO_P_ON) : uber_prototype[i]) ;
//    printf("Proto %d: ", a);
    printVector(prototype[a], N_FEATURES) ;
  }

  // load and pretrain each agent
  // multiagent-common.tcl contains everything agents should have in common
  printf("PRETRAINING:\n");
  for (a = 0 ; a < N_AGENTS ; a++) {
    lens("addNet agent%d %d %d %d", a, N_FEATURES, N_HIDDEN, N_FEATURES);
    lens("source multiagent-common.tcl") ;
    for (e = 0 ; e < PRETRAIN_EPOCHS ; e++) {
      // distort agent-specific prototype to create examplar
      // (could just load prototype and use "corruptInput" procedure [defined in
      //  multiagent-common.tcl] to distort input on-the-fly)
      for (i = 0 ; i < N_FEATURES ; i++)
	exemplar[i] = (WITH_PROB(ITEM_P_FLIP) ? WITH_PROB(PROTO_P_ON) : prototype[a][i]) ;
      // load exemplar (as both inputs and targets) 
      createLoadExample(exemplar, exemplar) ;
      lens("train 1");
    }
    // save last set of outputs
    saveOutputs(outputs[a]) ;
//    printf("Agent %d: ", a) ;
    printVector(outputs[a], N_FEATURES) ;
  }
  // for some number of iterations, select FROM and TO randomly, then
  // train TO on last output of FROM (saved in outputs[FROM])
  printf("COMMUNICATION:\n");
  for (e = 0 ; e < COMMUNICATION_ITERATIONS ; e++) {
    int from = RAND_INT(N_AGENTS) ;
    int to   = RAND_INT(N_AGENTS) ;
    lens("useNet agent%d", to) ;
    createLoadExample(outputs[from], outputs[from]) ;
    lens("train 1");
    saveOutputs(outputs[to]) ;
  }
  // print stuff
  for (a = 0 ; a < N_AGENTS ; a++) {
//    printf("Agent %d: ", a) ;
    printVector(outputs[a], N_FEATURES) ;
  }

  lens("deleteNets *"); // delete all networks because they will be recreated on next run
  printf("\nrunNum %d completed\n\n", runNum);

}

  return 0;
}

