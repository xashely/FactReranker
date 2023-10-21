export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ace14856qn/miniconda3/lib
CUDA_VISIBLE_DEVICES=0,1,2,3 python st_scorer.py --model_name_or_path all-MiniLM-L6-v2 --dataset_name /scratch/ace14856qn/scorer_pair_dataset --output_dir /scratch/ace14856qn/st_scorer_results/\
    --overwrite_output_dir --num_train_epochs 5 --do_train --eval_steps 1000 --do_eval
