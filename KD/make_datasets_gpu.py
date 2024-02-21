import datasets
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import time
import pandas as pd

# teacher 모델 및 토크나이저 불러오기
teacher_checkpoint = "medicalai/ClinicalBERT"
t_model = AutoModelForMaskedLM.from_pretrained(teacher_checkpoint)
t_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)

# 데이터 불러오기
dataset = datasets.load_from_disk('datasets_for_teacher')
dataset = dataset.select(list(range(4000000, 4417713)))
ds = dataset.to_pandas()

# 모델을 GPU로 이동
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t_model.to(device)

start_time = time.time()
for i in range(len(dataset)):
    if ds.loc[i,'mask_idx'] != -1:
        # MASK 위치 인덱스 가져오기
        tokenized = t_tokenizer(ds.loc[i,'text'], return_tensors='pt', add_special_tokens=False).to(device)
        mask = torch.where(tokenized['input_ids'][0] == 103)

        # top5 예측의 score, index
        top5 = torch.topk(torch.softmax(t_model(**tokenized).logits[0][mask], dim=1), 5)
        scores = top5.values[0].detach().cpu().numpy()
        indices = top5.indices[0]

        # top5 token decode
        answers = t_tokenizer.convert_ids_to_tokens(indices)

        # ds에 기록하기
        for j in range(5):
            s_col = 'score'+str(j)
            a_col = 'answer'+str(j)
            ds.loc[i,s_col] = scores[j]
            ds.loc[i,a_col] = answers[j]

    if i%1000 == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds for {i} rows")
        
        if i%10000 == 0:
            # GPU 캐시 비우기
            torch.cuda.empty_cache()

        # 중간저장
        if i%200000==0 and i>0:
            file = 'datasets_by_teacher_'+str(i+4000000)+'.csv'
            ds.to_csv(file, index=False)

# 데이터셋 저장
ds.to_csv('datasets_by_teacher_4.4m.csv', index=False)