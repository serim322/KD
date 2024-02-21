import numpy as np
from transformers import AutoModelForMaskedLM
import datasets
import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import collections
import numpy as np
from transformers import default_data_collator


# student2 모델 및 토크나이저 불러오기
#stu_checkpoint = "monologg/distilkobert"
stu_checkpoint = "/home/s1/serimkim/hf/distil-kobert-finetuned_KD_epoch4/checkpoint-15500"
stu_model = AutoModelForMaskedLM.from_pretrained(stu_checkpoint)
stu_tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")


# 전처리되어 저장된 datadict 불러오기
lm_datasets = datasets.load_from_disk("datasets_KD")

# 훈련-테스트 데이터셋 분리
test_size = 8833127 // 10   # dataset row : 8833127
train_size = 8833127 - test_size
downsampled_dataset = lm_datasets.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# 정해진 위치(mask_idx)의 토큰들에 적용하는 콜레이터 정의
def whole_word_masking_data_collator_with_padding(features, tokenizer):
    max_length = max(len(feature["input_ids"]) for feature in features)
    for feature in features:
        word_ids = feature.pop("word_ids")
        _ = feature.pop("score")

        # 단어와 해당 토큰 인덱스 간의 map 생성
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:    # word_id: 단어의 정수표현
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # 지정된 위치의 단어 마스킹
        mask = [feature.pop("mask_idx")]
        input_ids = feature["input_ids"]  # 인코딩된 input
        labels = feature["labels"]        # 정답 labels
        new_labels = [-100] * len(labels)
        for word_id in mask:  # mask배열에서 값이 0이 아닌 위치의 index return
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id


        # 패딩 적용
        padding_length = max_length - len(input_ids)
        feature['input_ids'] = input_ids + [tokenizer.pad_token_id] * padding_length
        feature['labels'] = new_labels + [-100] * padding_length

    return default_data_collator(features)


# training argument 설정
batch_size = 128
logging_steps = len(downsampled_dataset["train"]) // batch_size # Show the training loss with every epoch
model_name = "distil-kobert"
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned_KD_epoch5",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    remove_unused_columns=False,
    logging_steps=logging_steps,
    report_to="wandb"
)

# 트레이너 설정
trainer = Trainer(
    model=stu_model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=lambda features: whole_word_masking_data_collator_with_padding(features, stu_tokenizer)
)

# device 확인
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

# 모델 train
trainer.train()

# log_history
print(trainer.state.log_history)
