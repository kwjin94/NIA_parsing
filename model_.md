# 모델 설명서<br/><br/>


## 모델 description
- 모델명 AGRNet(Adaptive graph representation learning and reasoning for face parsing)<br/>
![image01](https://user-images.githubusercontent.com/112538268/209248821-fefa894f-77e9-4781-ae87-6959c86df654.png)<br/>
사진 [The  overview  of  the  proposed  face  parsing  framework]<br/>

Adaptive Graph Projection은 정점으로 구성 요소 특징을 추출하고, graph reasoning은 GCN(graph convolutional networks)을 사용하여 정점 간의 관계를 추론하고, graph reprojection은 학습된 그래프 표현을 다시 픽셀 그리드에 투영한다.<br/>
그래프 표현(graph representation)을 학습하여 영역별 관계를 모델링 및 추론하고 정밀한 분할을 위해 에지 정보를 활용하고 강조하는 모델이다. 일반적으로 CNN 기반 segmentation 모델은 인접한 픽셀 간의 정보만 활용하고, 멀리 떨어져 있는 영역과의 상관관계는 활용하지 않아서, 눈, 코, 입과 같은 얼굴의 구성요소 간 상호작용을 파악하는데 어려움이 있다. 하지만 그래프 표현을 사용하여 이러한 관계를 파악하고 이것이 얼굴 표현의 중요한 단서가 된다(예: 웃는 얼굴에서는 눈, 입, 눈썹에 곡선이 많아짐).<br/>
이미지 특징 추출은 ResNet-101을 일부 개조한 백본 네트워크를 통해 추출함. 그리고 context 정보를 최대한 활용하기 위해 피라미드 풀링 모듈로 멀티 스케일 특징을 추출한다.<br/>

## 모델 아키텍쳐
<img width="639" alt="image" src="https://user-images.githubusercontent.com/112538268/209280304-5410c669-0ece-4216-bd0e-545761e7bb5f.png">
사진 [Architecture  of  the  proposed  adaptive  graph  representation  learning  and  reasoning  for  face  parsing]<br/>  
<br/>
Backbone으로 ResNet-101을 수정하여 사용했다. Conv2, Conv5 layer의 output을 multi-scale representations를 위한 낮은 레벨과 높은 레벨 피처맵으로써 추출했다. Spatial sapce의 information loss를 줄이기 위해서, Dilated 컨볼루션을 사용하여 마지막 두개 block의 크기를 줄였다. Conv2, Conv3, Conv4의 출력으로 edge map을 예측했다.


## Input
- (3, 512, 512)
## Output
- (19, 512, 512)

## task
 - Segmentation
 
## training dataset
NIA dataset (150,000)<br/>
80% used for training (120,000 source images, grayscale images, edge images)<br/>
10% used for validation (15,000 source images, grayscale images, edge images)<br/>
10% used for testing (15,000 source images, grayscale images, edge images)<br/>


## training 요소들
- loss function
- cross entropy loss를 기본으로 쓴다.
- Boundary-Attention loss (BA-loss) : cross entropy loss에서 edge 부분의 loss를 보기 위해 iverson bracket을 추가해준 loss이다.
- optimizer : Stochastic Gradient Descent (SGD)
- epoch : 30
- learinng rate : 0.003
- batch size : 7

## evaluation metric
- 검증 목표 : F1 score = 70
- 달성 수치 : F1 score = 79.76


## 저작권
????
