model_dir='data/train_dev_conv_sum/chat_summary_combined_v7/train_issue_solution_20210702/models/bart-large-only-b64_6epochs_8gpus/'
python convert_pl_checkpoint_to_hf.py --pl_ckpt_path $model_dir/val_avg_rouge2=43.3202-step_count=20.ckpt \
--hf_src_model_dir $model_dir/best_tfmr --save_path $model_dir/ckpt20