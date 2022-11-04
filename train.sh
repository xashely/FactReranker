CUDA_VISIBLE_DEVICES=0 python run_summarization.py --model_name_or_path facebook/bart-base --test_file /scratch/ace14856qn/mimic/test_mimic.json --train_file /scratch/ace14856qn/mimic/train_mimic.json --validation_file /scratch/ace14856qn/mimic/test_mimic.json --output_dir /scratch/ace14856qn/results/\
    --per_device_train_batch_size=60\
    --per_device_eval_batch_size=60\
    --predict_with_generate --learning_rate 4e-07 --text_column findings --summary_column impression --preprocessing_num_workers 8 --do_predict --evaluation_strategy steps --num_train_epochs 5 --max_steps 40000  
