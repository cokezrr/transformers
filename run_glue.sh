export GLUE_DIR=/media/zrr/d5152fa4-8df8-4ccf-a396-fa62c0b68183/GLUE
export TASK_NAME=RTE

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 128 \
  --learning_rate 0.1 \
  --num_train_epochs 15 \
  --output_dir /home/zrr/ZRR/transformers/results/MNLI_base_cls_train_ep15_0.1 \
  --overwrite_output_dir \
  --eval_all_checkpoints \
  --save_steps 4000


