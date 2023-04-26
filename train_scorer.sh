export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ace14856qn/miniconda3/lib
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_scorer.py --model_name_or_path sentence-transformers/stsb-roberta-base-v2 --do_predict --dataset_name /scratch/ace14856qn/scorer_dataset --output_dir /scratch/ace14856qn/scorer_results/\
    --per_device_train_batch_size=75\
    --per_device_eval_batch_size=75\
    --overwrite_output_dir --do_predict --evaluation_strategy steps --num_train_epochs 5 --do_train --report_to wandb --eval_steps 500 
