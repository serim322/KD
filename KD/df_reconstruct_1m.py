import pandas as pd
import time

# 3군데 변경!! -> read_csv, file, to_csv

df = pd.read_csv('datasets_by_teacher_1m.csv')

columns = ['new_text', 'mask_idx', 'score']  # 열 이름 리스트
new_df2 = pd.DataFrame(columns=columns)
l = [300000, 600000]
start_time = time.time()

for i in range(len(df)):
    if i in l:
        file = 'dataset_0.'+str(i//100000)+'m.csv'
        new_df2.to_csv(file, index=False)
        columns = ['new_text', 'mask_idx', 'score']  # 열 이름 리스트
        new_df2 = pd.DataFrame(columns=columns)
    if df.loc[i, 'mask_idx'] != -1 :
        # 새로운 행 추가 - append() 메서드 사용
        new_data= pd.DataFrame({'new_text': [df.loc[i, 'text'].replace('[MASK]', str(df.loc[i, 'answer0'])),
                                               df.loc[i, 'text'].replace('[MASK]', str(df.loc[i, 'answer1']))],
                    'mask_idx': [df.loc[i, 'mask_idx'],
                                 df.loc[i, 'mask_idx']],
                     'score': [df.loc[i, 'score0'],
                               df.loc[i, 'score1']]})
        new_df2 = pd.concat([new_df2, new_data])
    else:
        data = pd.DataFrame({'new_text': [df.loc[i, 'text']],
                    'mask_idx': [-1],
                     'score': [1]})
        new_df2 = pd.concat([new_df2, data])
    if i%100000 == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds for {i} rows")

# 데이터셋 저장
new_df2.to_csv('dataset_1m.csv', index=False)