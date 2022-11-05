CUDA_VISIBLE_DEVICES=0,1,2,3 python run_summarization.py --model_name_or_path facebook/bart-base --do_train --do_predict --test_file /scratch/ace14856qn/mimic/test_mimic.json --train_file /scratch/ace14856qn/mimic/train_mimic.json --validation_file /scratch/ace14856qn/mimic/valid_mimic.json --output_dir /scratch/ace14856qn/results/\
    --per_device_train_batch_size=20\
    --per_device_eval_batch_size=20\
    --predict_with_generate --overwrite_output_dir --text_column findings --summary_column impression --preprocessing_num_workers 1 --do_predict --evaluation_strategy steps --num_train_epochs 10 --eval_steps 500 
