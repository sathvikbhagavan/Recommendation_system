@echo off

set dataset=EUR-Lex
set data_dir=../Sandbox/Data/%dataset%
set results_dir=../Sandbox/Results/%dataset%
set model_dir=../Sandbox/Results/%dataset%/model

set trn_ft_file=%data_dir%/trn_X_Xf.txt
set trn_lbl_file=%data_dir%/trn_X_Y.txt
set tst_ft_file=%data_dir%/tst_X_Xf.txt
set tst_lbl_file=%data_dir%/tst_X_Y.txt
set score_file=%results_dir%/score_mat.txt

:: training
:: Reads training features (in %trn_ft_file%), training labels (in %trn_lbl_file%), and writes FastXML model (to %model_dir%)
fastXML_train %trn_ft_file% %trn_lbl_file% %model_dir% -T 5 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10

:: testing
:: Reads test features (in %tst_ft_file%), FastXML model (in %model_dir%), and writes test label scores (to %score_file%)
fastXML_predict %tst_ft_file% %score_file% %model_dir%

:: performance evaluation
matlab -nodesktop -nodisplay -r "addpath(genpath('../Tools')); trn_X_Y = read_text_mat('%trn_lbl_file%'); tst_X_Y = read_text_mat('%tst_lbl_file%'); wts = inv_propensity(trn_X_Y,0.55,1.5); score_mat = read_text_mat('%score_file%'); get_all_metrics(score_mat, tst_X_Y, wts);"
