export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ace14856qn/miniconda3/lib
CUDA_VISIBLE_DEVICES=0,1,2,3 python st_entity_scorer.py --model_name_or_path /scratch/ace14856qn/results/ --dataset_name /scratch/ace14856qn/scorer_pair_process_dataset --test_dataset_name /scratch/ace14856qn/scorer_true_pair_dataset --output_dir /scratch/ace14856qn/cross_entity_scorer_results/\
    --per_device_train_batch_size=32\
    --per_device_eval_batch_size=512\
    --remove_unused_columns=False\
    --overwrite_output_dir --num_train_epochs 5 --do_train --do_eval --evaluation_strategy steps --dataloader_num_workers 32 --eval_steps 1000 --save_steps 100000 --warmup_ratio 0.1 --lr_scheduler_type constant_with_warmup --learning_rate 1e-5
