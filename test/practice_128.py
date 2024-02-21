import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMaskedLM, AdamW, get_scheduler
import datasets
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm.auto import tqdm
from transformers import pipeline
from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments
from transformers import Trainer


# student2 모델 및 토크나이저 불러오기
stu_checkpoint = "monologg/distilkobert"
stu_model = AutoModelForMaskedLM.from_pretrained(stu_checkpoint)
stu_tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
stu_makes_answers_of = pipeline(task='fill-mask', model=stu_model, tokenizer=stu_tokenizer)


# 전처리되어 저장된 datadict 불러오기
lm_datasets = datasets.load_from_disk("lm_datasets_len_128")


# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(tokenizer=stu_tokenizer, mlm_probability=0.10)


# 훈련-테스트 데이터셋 분리
test_size = 511485 // 10   # dataset row : 511485
train_size = 511485 - test_size
downsampled_dataset = lm_datasets.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


# training argument 설정
batch_size = 32
logging_steps = len(downsampled_dataset["train"]) // batch_size # Show the training loss with every epoch
model_name = "distil-kobert"
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-clinical-len128-1205",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
    report_to="wandb"
)

# 트레이너 설정
trainer = Trainer(
    model=stu_model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)

# device 확인
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

# 모델 train
trainer.train()

# log_history
print(trainer.state.log_history)
