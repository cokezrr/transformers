# SoT: Delving Deeper into Classification Head for Transformer

https://arxiv.org/pdf/2104.10935

In this paper, we propose a novel second-order transformer (SoT) model. The key of our SoT is a novel classification head which exploits simultaneously word tokens and classification token. The proposed classification head is flexible and fits for a variety of vision transformer architectures, significantly improving them on challenging image classification tasks. 

What’s more, the proposed classification head generalizes to language transformer architecture, performing much better than the conventional classification head on general language understanding tasks. 

This project provides the source code for text classification section in the SoT paper.

# Finetuning BERT with MGCrP module on GLUE tasks

## Pre-trained models

Model | Download
---|---
`bert-base-uncased` | [bert.base.uncased.bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin)
`bert-large-uncased` | [bert.large.uncased.bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin)
`bert-base-cased` | [bert.base.cased.bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin)
`bert-large-cased` | [bert.base.cased.bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin)

**Note:**
You don't need to download above models before fine-tune on GLUE tasks, as when you run code for the first time, code will automatically download those models for you.

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
# wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
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

For BERT-base model on each of the GLUE task, if only 4 gpus are available, you will need to use following cmd-line arguments:

Args | MNLI | QNLI | RTE | CoLA 
---|---|---|---|---
`--learning_rate` | 2e-5 | 3e-5 | 2e-5 | 2e-5 
`--per_gpu_train_batch_size` | 16 | 16 | 16 | 16
`--num_train_epochs` | 10 | 10 | 10 | 10

Meanwhile, you should change following values of [configuration_bert.py](transformers/configuration_bert.py) to adjust parameters in MGCrP module:

Config | MNLI | QNLI | RTE | CoLA 
---|---|---|---|---
`self.MGCrP_q` | 112 | 112 | 56 | 96
`self.MGCrP_k` | 48 | 48 | 32 | 48
`self.MGCrP_dp` | 0.7 | 0.5 | 0.8 | 0.5



