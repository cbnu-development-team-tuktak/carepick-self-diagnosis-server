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
- **사용한 Optimizer**: AdamW
- **손실 함수**: CrossEntropyLoss
- **토크나이저**: HuggingFace BertTokenizer
- **모델 저장 형식**: PyTorch `.pt` (model_staet_dit + LabelEncoder 포함)
  
### 🩺 진료과 예측 모델
- **사전학습 모델**: madatnlp/km-bert (한국어 특화 BERT)
- **모델 구조**: BERT + Linear classifier (transformers.AutoModelForSequenceClassification)
- **분류 대상 클래스 수**: 34개
- **출력 방식**: softmax 확률 기반 top-k 예측 결과 반환
- **학습 에폭 수**: 5 epochs
- **사용한 Optimizer**: Adamw
- **손실 함수**: CrossEntropyLoss
- **토크나이저**: HuggingFace AutoTokenizer
- **모델 저장 형식**: HuggingFace save_model 디렉토리 (`config.json`, `pytorch_model.bin`, `tokenizer.json` 등 포함)

## ⭐ 주요 기능
1. **질병 예측 API**
  - 사용자의 증상 문장을 입력 받아, 사전 학습된 BERT 기반 질병 분류 모델을 통해 예측
  - softmax 확률을 기반으로 가장 가능성 높은 질병 Top-K를 반환
2. **진료과 예측 API**
  - 사용자의 증상 문장을 입력 받아, 사전 학습된 BERT 기반 진료과 분류 모델을 통해 예측
  - softmax 확률을 기반으로 가장 가능성 높은 진료과 Top-K를 반환

## 👨‍💻 내 역할
진료과 예측 모델 업로드를 제외한 작업을 수행하였습니다.

- Flask 기반 서버 구조 설계 및 구현
- Blueprint를 활용한 라우팅 구조 모듈화 (`/disease`, `/specialty`)
- 사전학습 모델(`madatnlp/km-bert`)을 활용한 질병 분류 모델 학습 및 추론 로직 개발
- PyTorch `.pt` 체크포인트 구성 (model_state_dict + LabelEncoder) 및 로딩 처리
- `transformers`, `scikit-learn`, `torch` 기반의 추론 유틸리티 함수 개발
- 질병 예측 API (`/disease/mini`) 라우트 구성

## 🚀 실행 방법

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-id/carepick-self-diagnosis.git
cd carepick-self-diagnosis

# 2. 가상환경 설정 (선택)
python3 -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 모델 파일(.pt 또는 HuggingFace 디렉토리) 위치 확인
# 예시: ./models/disease_classifier.pt, ./models/specialty_model/

# 5. 서버 실행
python app.py
```
