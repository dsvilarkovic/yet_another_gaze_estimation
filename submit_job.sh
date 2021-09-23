#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -l gpu
#$ -l h_vmem=40G
#$ -q gpu.24h.q@*
#$ -l hostname='biwirender0[56789]|biwirender1[0123457]|bmicgpu0[1-5]'
#$ -j y
#$ -o /scratch_net/snapo_second/nipopovic/workspace/mp_project/Output/cluster_logs
#$ -e /scratch_net/snapo_second/nipopovic/workspace/mp_project/Output/cluster_logs

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Set Pathsa
# PROJECT_ROOT_DIR=/scratch_net/snapo_second/nipopovic/workspace/specta/Disease_classification_PROJECT
# export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT_DIR}
# cd ${PROJECT_ROOT_DIR}
# pwd


OMP_NUM_THREADS=8
/scratch_net/snapo_second/nipopovic/apps/miniconda3/envs/new_specta/bin/python /scratch_net/snapo_second/nipopovic/workspace/mp_project/Cyclopes/running.py # ${@:2}
#/scratch_net/snapo_second/nipopovic/apps/miniconda3/envs/new_specta/bin/python main.py ${@:2}
