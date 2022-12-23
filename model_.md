# 모델 설명서<br/><br/>


## 모델 description
- 모델명 AGRNet(Adaptive graph representation learning and reasoning for face parsing)<br/>

    Adaptive Graph Projection은 정점으로 구성 요소 특징을 추출, graph reasoning은 GCN을 사용하여 정점 간의 관계를 추론, graph reprjection은 학습된 그래프 표현을 다시 픽셀 그리드에 투영.
그래프 표현(graph representation)을 학습하여 영역별 관계를 모델링 및 추론하고 정밀한 분할(segmentation)을 위해 에지 정보를 활용하고 강조하는 모델임. 일반적으로 CNN 기반 분할 모델은 인접한 픽셀 간의 정보만 활용하고, 멀리 떨어져 있는 영역과의 상관관계는 활용하지 않음. 특히, 눈, 입, 코와 같은 얼굴의 구성요소 간 상호작용을 파악하기 어렵지만 이러한 관계는 얼굴 표현의 중요한 단서임(예: 웃는 얼굴에서는 눈, 입, 눈썹에 곡선이 많아짐).
얼굴 영역 간의 관계를 모델링하고 장거리 상관관계를 캡처하기 위해 그래프 표현을 학습함. 즉, 얼굴 이미지의 픽셀을 그래프 구조로 표현함. 위 그림과 같이 facial landmark를 그래프로 표현할 수 있음. 
그래프 표현 학습에서 에지 정보를 강조하여 서로 다른 구성요소 사이의 에지를 따라 정확한 분할을 유도함. 강조는 에지 픽셀의 특징에 더 큰 가중치를 할당하는 것임.
특징 및 에지 추출은 ResNet을 일부 개조한 백본 네트워크로 이미지의 특징을 추출함. 그리고 context 정보를 최대한 활용하기 위해 피라미드 풀링 모듈로 멀티 스케일 특징을 추출함.(PSP의 PPM, DeepLab의 ASPP 참조) 마지막으로 에지맵을 얻기 위해 에지 인식 모듈을 구성함.






## 모델 아키텍쳐

## Input

## Output

## task

## training dataset

## training 요소들

## evaluation metric

## 저작권
