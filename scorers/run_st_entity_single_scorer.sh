export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ace14856qn/miniconda3/lib
CUDA_VISIBLE_DEVICES=0,1,2,3 python st_entity_single_scorer.py --model_name_or_path /scratch/ace14856qn/openi_results/ --dataset_name /scratch/ace14856qn/scorer_process_openi_pair_dataset --test_dataset_name /scratch/ace14856qn/scorer_true_openi_pair_dataset --output_dir /scratch/ace14856qn/cross_entity_scorer_openi_results/\
    --per_device_train_batch_size=4\
    --per_device_eval_batch_size=4\
    --remove_unused_columns=False\
    --overwrite_output_dir --num_train_epochs 10 --do_train --do_eval --evaluation_strategy steps --dataloader_num_workers 32 --eval_steps 1 --save_steps 1 --cache_dir /scratch/ace14856qn/cache_openi_results --do_predict
