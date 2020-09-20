#!/bin/bash

dataset="dataset"
data_dir="$dataset"
results_dir="$dataset"
model_dir="$dataset/model"

trn_ft_file="${data_dir}/mydata.train.X"
trn_lbl_file="${data_dir}/mydata.train.y"
tst_ft_file="${data_dir}/mydata.test.X"
tst_lbl_file="${data_dir}/mydata.test.y"
score_file="${results_dir}/score_mat.txt"

# create the model folder
mkdir -p $model_dir

# training
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), and writes FastXML model (to $model_dir)
./fastXML_train $trn_ft_file $trn_lbl_file $model_dir -T 5 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10

# testing
# Reads test features (in $tst_ft_file), FastXML model (in $model_dir), and writes test label scores (to $score_file)
./fastXML_predict $tst_ft_file $score_file $model_dir


