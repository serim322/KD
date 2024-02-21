import datasets
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# teacher 모델 및 토크나이저 불러오기
teacher_checkpoint = "medicalai/ClinicalBERT"
t_model = AutoModelForMaskedLM.from_pretrained(teacher_checkpoint)
t_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)

# 데이터 불러오기
datasets_len10_mask = datasets.load_from_disk('datasets_for_teacher')

# dataset 구성 함수
def make_answers(examples):
    if examples['mask_idx'] != -1:
        # MASK 위치 인덱스 가져오기
        tokenized = t_tokenizer(examples['text'], return_tensors='pt', add_special_tokens=False)
        mask = torch.where(tokenized['input_ids'][0] == 103)

        # top5 예측의 score, index
        top5 = torch.topk(torch.softmax(t_model(**tokenized).logits[0][mask], dim=1), 5)
        examples['scores'] = top5.values[0]
        indices = top5.indices[0]

        # top5 token decode
        examples['answers'] = t_tokenizer.convert_ids_to_tokens(indices)

    else:
        examples['scores'] = []
        examples['answers'] = []
    return examples

# 답변 생성
datasets_by_teacher = datasets_len10_mask.map(make_answers)

# 데이터셋 저장
datasets_by_teacher.save_to_disk("./datasets_by_teacher")  # 저장 경로명 변경
