# NIA_parsing
### inplace_abn
Also, we use In-Place Activated BatchNorm.   
First, you need to clone and compile inplace_abn.
버전 호환 문제로 requirements.txt 수정필요.

```
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
cd scripts
pip install -r requirements.txt
```
   
## 1. Load docker container
>docker run --gpus all -d -it --name nia_torch_gpu --ipc=host -v "$(pwd)":/EAGR nvcr.io/nvidia/pytorch:22.02-py3 bash   
>>python == 3.8.12   
torch == 1.11.0   
torchvision == 0.12.0
  
## 2. Check data
>check.py -> check.txt
- check.txt 구성
  - 전체 데이터 수
  - 106 point 수
  - bbox 데이터 수
  - class 별 수 및 class 당 이미지 수
  
#### Change 16bit image to 8 bit
> bit_change.ipynb
- labels 이미지를 16비트로 주면 8비트로 변환하고 진행해야한다.
- 추후에 train_NIA.py에서 자체적으로 이미지 읽을때 8비트 변환하고
- evaluate.py 에서도 8비트 변환하게 해주면 미리 16 to 8 bit로 변환 해 줄 필요 없다.
## 3. Preprocess 
> preprocess.ipynb
- make (image, json) list, crop, split train/test
- 원본 이미지를 512 x 512 사이즈로 crop 한다.
- Crop된 이미지를 train set과 test set으로 분리하여 저장한다.
## 4. Make list
> make_list_nia.ipynb
- (3.) 에서 만든 폴더에 저장된 이미지 리스트를 txt 파일로 저장한다.
## 5. Generate edge
> generate_edge_NIA.py
- (3.) 에서 만든 폴더에 저장된 lables 이미지를 가지고 edge 이미지를 생성한다.
### 최종 산출물
>train 폴더
>> images : 원본 이미지   
>> labels : GT grayscale 이미지   
>> edges : GT edge 이미지   

>test 폴더
>> images : 원본 이미지   
>> labels : GT grayscale 이미지   
>> edges : GT edge 이미지  

>check.txt   
train_id.txt   
test_id.txt   
test_labels.txt : class 별 id   

## Train
> train_NIA.py   

## Test
> evaluate.py
