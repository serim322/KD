from kobert_tokenizer import KoBERTTokenizer
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import torch


# 데이터셋 로드
MACCROBAT = load_dataset("singh-aditya/MACCROBAT_biomedical_ner")
label_list = MACCROBAT["train"].features["ner_labels"].feature.names

# label, id dict
id2label = {}
label2id = {}
for idx, label in enumerate(label_list):
    id2label[idx] = label
    label2id[label] = idx

# student 토크나이저 불러오기
student_checkpoint = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(student_checkpoint)

# student 모델 불러오기
model = AutoModelForTokenClassification.from_pretrained(
    "/home/s1/serimkim/hf/distil-kobert-finetuned_KD_epoch4/checkpoint-15500", num_labels=83, id2label=id2label, label2id=label2id
)

# 토크나이징 함수
def tokenize_words(example):
    tokenized_inputs = tokenizer(example["tokens"],is_split_into_words=True)

    labels = [-100]
    for i, tokens in enumerate(example['tokens']):
        tokenized_ids = tokenizer(tokens, add_special_tokens=False)['input_ids']
        if len(tokenized_ids) != 0:
            label_ids = [-100] * len(tokenized_ids)
            label_ids[0] = example['ner_labels'][i]
            labels += label_ids
    labels.append(-100)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# 각 예제에 대해 'words'를 토크나이징하여 'word_id' 생성
tokenized_MACCROBAT = MACCROBAT.map(tokenize_words)

# 데이터셋 분할
tokenized_MACCROBAT = tokenized_MACCROBAT["train"].train_test_split(test_size=0.2)

# 불필요한 컬럼 삭제
tokenized_MACCROBAT = tokenized_MACCROBAT.remove_columns(['full_text', 'ner_info', 'tokens', 'ner_labels', 'token_type_ids', 'attention_mask'])

# 동일한 길이로 변경
chunk_size = 256
def group_texts(examples):
    # 모든 텍스트들을 결합한다.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 결합된 텍스트들에 대한 길이를 구한다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # `chunk_size`보다 작은 경우 마지막 청크를 삭제
    total_length = (total_length // chunk_size) * chunk_size
    # max_len 길이를 가지는 chunk 단위로 슬라이스
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    return result
tokenized_MACCROBAT = tokenized_MACCROBAT.map(group_texts, batched=True)

# 콜레이터 정의
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 평가 준비
seqeval = evaluate.load("seqeval")

# 평가 함수 정의
example = MACCROBAT['train'][0]
labels = [label_list[i] for i in example[f"ner_labels"]]
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 학습 준비
batch_size = 32
logging_steps = len(tokenized_MACCROBAT["train"]) // batch_size
training_args = TrainingArguments(
    output_dir="finetuned_distilkobert_ner_10",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=logging_steps,
    report_to="wandb"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_MACCROBAT["train"],
    eval_dataset=tokenized_MACCROBAT["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# device 확인
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

# 모델 train
trainer.train()

# log_history
print(trainer.state.log_history)
