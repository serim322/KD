import json
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import random
import time


# 데이터셋 불러오기
raw_datasets = load_dataset('starmpcc/Asclepius-Synthetic-Clinical-Notes', 'train')
raw_datasets

# 판다스 데이터프레임으로 변환
dataset = pd.DataFrame(raw_datasets['train'].to_pandas())
dataset

# 추가 컬럼 생성
dataset['korean'] = ""
dataset['mixed'] = ""
dataset['response'] = ""

# prompt에 넘겨줄 예제 불러오기
notes = pd.read_excel('notes.xlsx')
samples = notes[:50]
# prompt로 넘겨줄 예제 랜덤 샘플링하는 함수 정의


def make_prompt(input_text):
    # 0부터 49까지의 숫자 리스트 생성
    numbers = list(range(50))
    # 리스트에서 3개의 랜덤한 숫자 선택
    random_numbers = random.sample(numbers, 3)
    prompt = """For the given ‘english’ text, generate ‘korean’ and ‘mixed’ in json format.
‘Korean’ is a translation of ‘english’ into Korean, and ‘mixed’ is a translation of English and Korean mixed together.
<sample1> ~ <sample3> is a example of creating ‘korean’ and ‘mixed’ from ‘english’.
"""
    for i in [1,2,3]:
        prompt += f"\n\n<sample{i}>\nenglish: {samples['snippet'][i]}\n"
        prompt += f"{{\"korean\":\"{samples['korean'][i]}\", \"mixed\": \"{samples['mixed'][i]}\"}}"
    prompt += "\n\nAs in the examples above, write ‘korean’ and ‘mixed’ for ‘English’ below. please just return json.\n\n"
    prompt += input_text
    return prompt

# api 호출 함수 정의
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
)
def gpt_response(input_text):
    prompt = make_prompt(input_text)
    # OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a translator from English to a mixed text of English and Korean."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content
    return answer


start_time = time.time()
for i in range(400, 1000):
    response = gpt_response(dataset['note'][i].replace('\n', ''))
    print(i)
    try:
        response = json.loads(response.replace('\n', ''))
        dataset['korean'][i] = response.get('korean', 'Error: Korean data not found')
        dataset['mixed'][i] = response.get('mixed', 'Error: Mixed data not found')
        if i>400 and i % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Elapsed Time for {i} iterations: {elapsed_time} seconds")
            num = i
            gpt_data = dataset[i-100:i]
            gpt_data.to_excel(f'gpt_data_{i}.xlsx')
            start_time = elapsed_time
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for index {i}: {e}")
        # 에러 발생 시 스킵하고 다음 i에 대해 반복
        continue
    except KeyError as e:
        print(f"Error accessing key in response for index {i}: {e}")
        dataset['response'][i] = response
        # 에러 발생 시 스킵하고 다음 i에 대해 반복
        continue

print('===================== Augmentation is done. ====================')
gpt_data = dataset[900:1000]
gpt_data.to_excel(f'gpt_data_1000.xlsx')