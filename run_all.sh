#!/bin/bash

commands=(
  # Main Results
  "bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_FB15K_DB15K_0.2.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.5 0 0 > log_FB15K_DB15K_0.5.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.8 0 0 > log_FB15K_DB15K_0.8.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.2 128 0 > log_FB15K_YAGO15K_0.2.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.5 0 0 > log_FB15K_YAGO15K_0.5.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.8 0 0 > log_FB15K_YAGO15K_0.8.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.5 0 0 data/FB15K_DB15K_0.5_train_ill.npy > log_FB15K_DB15K_0.2.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.5 0 0 data/FB15K_DB15K_0.5_train_ill.npy > log_FB15K_DB15K_0.5.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.8 0 0 data/FB15K_DB15K_0.8_train_ill.npy > log_FB15K_DB15K_0.8.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.2 0 0 data/FB15K_YAGO15K_0.2_train_ill.npy > log_FB15K_YAGO15K_0.2.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.5 0 0 data/FB15K_YAGO15K_0.5_train_ill.npy > log_FB15K_YAGO15K_0.5.out 2>&1"
  "bash run.sh 42 FB15K_YAGO15K 0.8 0 0 data/FB15K_YAGO15K_0.8_train_ill.npy > log_FB15K_YAGO15K_0.8.out 2>&1"

  # Ablation Study
  "bash run.sh 42 FB15K_DB15K 0.2 0 1 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_1.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 2 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_2.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 3 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_3.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 4 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_4.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 5 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_5.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 6 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_6.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 7 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_single_align.out 2>&1"
  "bash run.sh 42 FB15K_DB15K 0.2 0 8 data/FB15K_DB15K_0.2_train_ill.npy > log_FB15K_DB15K_0.2_without_joint_aligh_single.out 2>&1"

  # Vary AL_LOSS only
  "AL_LOSS=0.01 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_al0.01.out 2>&1"
  "AL_LOSS=0.05 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_al0.05.out 2>&1"
  "AL_LOSS=0.1 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_al0.1.out 2>&1"
  "AL_LOSS=0.5 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_al0.5.out 2>&1"
  "AL_LOSS=1.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_al1.0.out 2>&1"

  # Vary CL_LOSS only
  "CL_LOSS=0.01 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_cl0.01.out 2>&1"
  "CL_LOSS=0.1 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_cl0.1.out 2>&1"
  "CL_LOSS=0.5 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_cl0.5.out 2>&1"
  "CL_LOSS=0.05 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_cl0.05.out 2>&1"
  "CL_LOSS=1.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_loss_cl1.0.out 2>&1"


  # Vary TAU_CL only
  "TAU_CL=0.01 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_cl_0.01.out 2>&1"
  "TAU_CL=0.05 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_cl_0.05.out 2>&1"
  "TAU_CL=0.1 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_cl_0.1.out 2>&1"
  "TAU_CL=0.5 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_cl_0.5.out 2>&1"
  "TAU_CL=1.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_cl_1.0.out 2>&1"

  # Vary TAU_AL only
  "TAU_AL=0.1 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_al_0.1.out 2>&1"
  "TAU_AL=1.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_al_1.0.out 2>&1"
  "TAU_AL=2.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_al_2.0.out 2>&1"
  "TAU_AL=4.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_al_4.0.out 2>&1"
  "TAU_AL=8.0 bash run.sh 42 FB15K_DB15K 0.2 512 0 > log_tau_al_8.0.out 2>&1"

)

is_gpu_idle() {
  ! nvidia-smi | grep "bin/python3" > /dev/null
}

for cmd in "${commands[@]}"; do
  echo "Preparing to run: $cmd"

  while ! is_gpu_idle; do
    echo "$(date) - GPU busy, waiting..."
    sleep 60
  done

  echo "$(date) - GPU idle. Launching command: $cmd"
  eval "$cmd"

  echo "$(date) - Finished: $cmd"
  sleep 60
done

echo "All jobs finished."
