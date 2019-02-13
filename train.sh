#!/usr/bin/env bash
USR_DIR=/Users/mike/CS6320/extracredit2
PROBLEM=reuters_nlp
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
OUTDIR=./trained_model-wider
rm -rf $OUTDIR
t2t-trainer \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$USR_DIR \
  --problem=$PROBLEM \
  --model=transformer \
  --hparams_set=transformer_reuters \
  --output_dir=$OUTDIR\
  --job-dir=$OUTDIR\
  --train_steps=3000 \
  --eval_steps=100
