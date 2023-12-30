---
title: Dataset & DataLoader
author: soyoonjeong
date: 2023-12-26 10:49:00 +0800
categories: [Naver Boostcamp, PyTorch]
tags: [네부캠, PyTorch, Dataset, DataLoader]
toc: true
commencts : true
math : true
---

# [네부캠 2주차] Dataset & DataLoader


![Untitled](%5B%E1%84%82%E1%85%A6%E1%84%87%E1%85%AE%E1%84%8F%E1%85%A2%E1%86%B7%202%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%5D%20Dataset%20&%20DataLoader%20536fc64221ee498e912ac9b21b001f22/Untitled.png)

## Dataset 클래스

- 데이터 입력 형태를 정의하는 클래스

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
	def __init__(self, text, labels): # 초기 데이터 생성 방법을 지정
		self.data = text
		self.labels = labels
	
	def __len__(self): # 데이터 전체 길이
		rerturn len(self.labels)

	def __getitem__(self, idx):
		label = self.labels[idx]
		text = self.data[idx]
		sample = {"Text" : text, "Class" : label}
		return sample

dataset_custom = CustomDataset()
```

- 기본적으로 Dataset을 구성할 때는 PyTorch의 torch.utils.data에서 Dataset 클래스를 상속하여 만듭니다.
- `__init__` 클래스는 데이터의 위치나 파일명과 같은 초기화 작업을 위해 동작합니다.
- `__len__` 클래스는 Dataset의 최대 요소 수를 반환합니다.
- `__getitem__` 클래스는 데이터셋의 idx 번째 데이터를 반환합니다. 이 클래스에서 원본 데이터를 가져와 전처리하고 증강하는 과정을 거칩니다.

## DataLoader 클래스

```python
DataLoader(dataset,            # Dataset 인스턴스가 들어감
           batch_size=1,       # 배치 사이즈를 설정
           shuffle=False,      # 데이터를 섞어서 사용하겠는지를 설정
           sampler=None,       # sampler는 index를 컨트롤
           batch_sampler=None, # 위와 비슷하므로 생략
           num_workers=0,      # 데이터를 불러올때 사용하는 서브 프로세스 개수
           collate_fn=None,    # map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능
           pin_memory=False,   # Tensor를 CUDA 고정 메모리에 할당
           drop_last=False,    # 마지막 batch를 사용 여부
           timeout=0,          # data를 불러오는데 제한시간
           worker_init_fn=None # 어떤 worker를 불러올 것인가를 리스트로 전달
          )

dataloader_custom = DataLoader(dataset_custom)
```

- 주어진 데이터셋을 미니 배치로 분할하고 미니 배치별로 학습을 할 수 있게 도와줍니다.
- 학습 직전 데이터의 변환을 책임집니다 (Tensor로 변환)
- `batch_size` : 배치 사이즈를 결정합니다.
- `shuffle` : 데이터를 섞어서 사용하겠는지를 결정합니다.
- `sampler` : index를 컨트롤하는 방법, index를 컨트롤하기 위해서 shuffle은 False여야 합니다.
    - SequentialSampler : 항상 같은 순서
    - RandomSampler : 랜덤, replacement 여부 선택 가능, 개수 선택 가능
    - SubsetRandomSampler : 랜덤 리스트
    - WeightRandomSampler : 가중치에 따른 확률
    - BatchSampler : batch 단위로 sampling 가능
    - DistributedSampler : 분산처리
    - 불균형 데이터셋의 경우, 클래스 비율에 맞게끔 데이터를 제공해야할 필요가 있음
- `num_workers` : 데이터를 불러올 때 사용하는 서브 프로세스 개수
    - 윈도우에서는 멀티프로세서의 제한 때문에 num worker의 수가 0 이상인 경우 에러가 발생합니다.
    - 무작정 num_workers의 수를 높인다고 좋지 않습니다! 데이터를 불러와 cpu와 gpu 사이의 교류가 많아진다면 오히려 병목이 생깁니다.
- `collate_fn` : sample list를 batch 단위로 바꾸기 위해 필요한 기능
    
    ![Untitled](%5B%E1%84%82%E1%85%A6%E1%84%87%E1%85%AE%E1%84%8F%E1%85%A2%E1%86%B7%202%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%5D%20Dataset%20&%20DataLoader%20536fc64221ee498e912ac9b21b001f22/Untitled%201.png)
    
    - 직관적인 예시로는 프린터기에서 인쇄할 때 묶어서 인쇄하기와 같은 기능이라고 생각하면 됩니다.
    - ((피처1, 라벨1), (피처2, 라벨2))와 같은 배치 단위 데이터가 ((피처1, 피처2), (라벨1, 라벨2))와 같이 바뀝니다.
    
    ```python
    class ExampleDataset(Dataset):
        def __init__(self, num):
            self.num = num
    
        def __len__(self):
            return self.num
    
        def __getitem__(self, idx):
            return {"X":torch.tensor([idx] * (idx+1), dtype=torch.float32),
                    "y": torch.tensor(idx, dtype=torch.float32)}
    
    #tensor([[0.]])
    #tensor([[1., 1.]])
    #tensor([[2., 2., 2.]])
    #tensor([[3., 3., 3., 3.]])
    #tensor([[4., 4., 4., 4., 4.]])
    #tensor([[5., 5., 5., 5., 5., 5.]])
    #tensor([[6., 6., 6., 6., 6., 6., 6.]])
    #tensor([[7., 7., 7., 7., 7., 7., 7., 7.]])
    #tensor([[8., 8., 8., 8., 8., 8., 8., 8., 8.]])
    #tensor([[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]])
    
    def my_collate_fn(samples):
        collate_X = []
        collate_y = []
    
        max_len = max([len(sample['X']) for sample in samples])
        for sample in samples:
            diff = max_len - len(sample['X'])
            if diff > 0:
                zero_pad = torch.zeros(diff)
                collate_X.append(torch.cat((sample['X'], zero_pad), 0))
            else:
                collate_X.append(sample['X'])
            collate_y.append(sample['y'])
    
        return {'X': torch.stack(collate_X),
                 'y': torch.stack(collate_y)}
    
    dataloader_example = torch.utils.data.DataLoader(dataset_example,
                                                     batch_size=2,
                                                     collate_fn=my_collate_fn)
    for d in dataloader_example:
        print(d['X'], d['y'])
    
    #tensor([[0., 0.],
    #        [1., 1.]]) tensor([0., 1.])
    #tensor([[2., 2., 2., 0.],
    #        [3., 3., 3., 3.]]) tensor([2., 3.])
    #tensor([[4., 4., 4., 4., 4., 0.],
    #        [5., 5., 5., 5., 5., 5.]]) tensor([4., 5.])
    #tensor([[6., 6., 6., 6., 6., 6., 6., 0.],
    #        [7., 7., 7., 7., 7., 7., 7., 7.]]) tensor([6., 7.])
    #tensor([[8., 8., 8., 8., 8., 8., 8., 8., 8., 0.],
    #        [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]]) tensor([8., 9.])
    ```
    
- `pin_memory` : Tensor를 CUDA 고정 메모리에 할당시켜 데이터 전송을 빠르게 합니다.
- `drop_last` : 마지막 batch 사용 여부를 결정합니다.
- `time_out` : 양수로 주어지는 경우 DataLoader가 data를 불러오는 데 제한시간입니다.

```python
text = ['Happy', 'Amazing', 'Sad', 'Unhappy', 'Glum']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
MyDataset = CustomDataset(text, labels)

MyDataloader = DataLoader(MyDataset, batch_size = 2, shuffle = True)
next(iter(MyDataloader))
# {‘Text’: ['Glum', ‘Sad'], 'Class': ['Negative', 'Negative’]}

for batch in MyDataloader:
	print(batch)
# {‘Text’: ['Glum', 'Unhappy'], 'Class': ['Negative', 'Negative’]}
# {‘Text’: [‘Sad', ‘Amazing'], 'Class': ['Negative', ‘Positive’]}
# {‘Text’: [‘happy'], 'Class': [‘Positive']}
```