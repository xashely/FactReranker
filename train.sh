CUDA_VISIBLE_DEVICES=0 python run_summarization.py --model_name_or_path facebook/bart-base --do_train --do_eval --test_file /scratch/ace14856qn/mimic/test_mimic.json --train_file /scratch/ace14856qn/mimic/train_mimic.json --validation_file /scratch/ace14856qn/mimic/valid_mimic.json --source_prefix "summarize: " --output_dir /scratch/ace14856qn/results/\
    --overwrite_output_dir \
    --per_device_train_batch_size=15\
    --per_device_eval_batch_size=15\
    --predict_with_generate --text_column findings --summary_column impression --max_target_length 50 --val_max_target_length 50 --preprocessing_num_workers 6 --do_predict
