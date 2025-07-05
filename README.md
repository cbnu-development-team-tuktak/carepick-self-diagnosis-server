# Carepick Self Diagnosis Server - 질병/진료과 예측 모델 서버

## 📄 프로젝트 소개
Carepick Self Diagnosis Server는 사용자의 증상 입력에 따라 사전 학습된 모델을 기반으로, <br>
질병 또는 진료과를 예측하여 Top-k 결과를 반환하는 Flask 기반 서버입니다. <br>

이 서버는 메인 백엔드(Spring Boot)와 분리되어 있어, <br>
**서버 부하를 분산**하고 **Python 기반의 사전 학습 모델 처리 효율성**을 극대화하기 위해 <br>
독립적으로 구성되어 있습니다.

## 🛠️ 가술 스택

### 📌 주요 프레임워크 및 언어
- **Python 3.10**: 모델 추론 및 서버 구현
- **Flask**: REST API 서버 프레임워크
- **Blueprint**: 예측 기능을 모듈 단위로 분리
  
### 🤖 머신러닝 / 딥러닝
- **PyTorch**: 모델 로딩, 추론 및 가중치 저장
- **transformers (HuggingFace)**: Bert tokenizer 및 분류기 사용
- **scikit-learn**: 클래스 디코딩용 LabelEncoder 제공
- **madatnlp/km-bert**: 한국어 특화 사전학습 BERT 모델
  
## 📊 모델 정보
### 🦠 질병 예측 모델
- **사전학습 모델**: madatnlp/km-bert (한국어 특화 BERT)
- **모델 구조**: BERT + Linear classifier (transformers.BertForSequenceClassification)
- **분류 대상 클래스 수**: 146개
- **출력 방식**: softmax 확률 기반 top-k 예측 결과 반환
- **학습 에폭 수**: 10 epochs
- **사용한 Optimizer**: AdamW (기본 설정 기준)
- **손실 함수**: CrossEntropyLoss
- **토크나이저**: HuggingFace BertTokenizer
- **모델 저장 형식**: PyTorch `.pt` (model_staet_dit + LabelEncoder 포함)
  
### 🩺 진료과 예측 모델
