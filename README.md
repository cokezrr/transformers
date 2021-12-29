# SoT: Delving Deeper into Classification Head for Transformer

https://arxiv.org/pdf/2104.10935

# Finetuning BERT with MGCrP module on GLUE tasks

## Results

**[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)**
_(dev set, single model, single-task finetuning)_

Model | CoLA | RTE | MNLI | QNLI 
---|---|---|---|---
`BERT-base+MGCrP` | 58.03 | 69.31 | 84.20 | 90.78 
`BERT-large+MGCrP` | 61.82 | 75.09 | 86.46 | 92.37 

## Example usage

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Fine-tuning on GLUE task:
Example fine-tuning BERT-base with MGCrP module cmd for `RTE` task
```bash
export GLUE_DIR=/path/to/GLUE
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
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir OUTPUT/ \
  --eval_all_checkpoints 
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | RTE | CoLA 
---|---|---|---|---
`--learning_rate` | 2e-5 | 3e-5 | 2e-5 | 2e-5 



