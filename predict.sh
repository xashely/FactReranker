export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ace14856qn/miniconda3/lib
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_summarization_with_scorer.py --model_name_or_path /scratch/ace14856qn/results/ --do_predict --test_file /scratch/ace14856qn/mimic/test_mimic.json --train_file /scratch/ace14856qn/mimic/train_mimic.json --validation_file /scratch/ace14856qn/mimic/valid_mimic.json --output_dir /scratch/ace14856qn/scorer_specified_results/\
    --per_device_train_batch_size=10\
    --per_device_eval_batch_size=8\
    --predict_with_generate --overwrite_output_dir --text_column findings --summary_column impression --preprocessing_num_workers 4 --do_predict --evaluation_strategy no --num_beams 10 --include_inputs_for_metrics --disable_tqdm False
