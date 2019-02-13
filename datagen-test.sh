#!/usr/bin/env bash
#Adapted from: https://tensorflow.github.io/tensor2tensor/new_problem.html
#Mike Holcomb
#Reuters 5-gram example for CS6320

USR_DIR=/Users/mike/CS6320/extracredit2
PROBLEM=reuters_nlp_test
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
