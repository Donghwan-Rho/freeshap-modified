#!/bin/sh
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate inv
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-6
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-5
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-4
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-3
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-2
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 1000 --lambda_ 1e-1
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-6
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-5
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-4
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-3
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-2
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 2000 --lambda_ 1e-1
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-6
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-5
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-4
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-3
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-2
python sst2_shapley_acc.py --num_train_dp 10000 --tmc_iter 500 --approximate eigen --eigen_rank 3000 --lambda_ 1e-1