# Carepick Flask Server

> 증상 기반 질병 및 진료과 예측 AI 추론 서버

## 프로젝트 개요

- **주제**: 사용자의 자연어 증상 입력을 받아 질병 및 진료과를 Top-K 형식으로 예측하는 Flask 기반 추론 서버
- **기간**: 2024.09.02 ~ 2025.12.11
- **연동**: Carepick Backend (Spring Boot) → Flask 서버 POST 요청

## 기술 스택

| 영역 | 기술 |
|------|------|
| Framework | Flask |
| 딥러닝 | PyTorch |
| 모델 | BERT (BertForSequenceClassification, AutoModelForSequenceClassification) |
| 토크나이저 | BertTokenizer, AutoTokenizer (Hugging Face Transformers) |
| 모델 저장소 | Hugging Face Hub |
| 배포 | AWS EC2 (포트 10000) |

## 프로젝트 구조

```
carepick-flask/
├── app.py                          # Flask 앱 진입점
├── routes/
│   ├── disease.py                  # /disease 엔드포인트
│   └── specialty.py                # /specialty 엔드포인트
├── model/
│   ├── disease_prediction_model.py # 질병 예측 모델 로딩
│   ├── specialty_prediction_model.py # 진료과 예측 모델 로딩
│   ├── loader.py                   # 공통 모델 로더
│   └── specialty_classifier/       # 진료과 모델 로컬 저장 디렉토리
│       └── label_classes.npy
└── inference/
    ├── disease_predictor.py        # 질병 Top-K 추론 함수
    └── specialty_predictor.py      # 진료과 Top-K 추론 함수
```

## API 명세

### POST /disease

사용자 증상 텍스트를 받아 질병 Top-K를 예측합니다.

**Request**
```json
{
  "text": "머리가 아프고 열이 나요"
}
```

**Query Parameter**
| 파라미터 | 타입 | 기본값 | 설명 |
|------|------|------|------|
| k | int | 3 | 반환할 Top-K 개수 |

**Response**
```json
[
  {"disease": "감기", "score": 0.7823},
  {"disease": "독감", "score": 0.1245},
  {"disease": "편도염", "score": 0.0612}
]
```

---

### POST /specialty

사용자 증상 텍스트를 받아 진료과 Top-K를 예측합니다.

**Request**
```json
{
  "text": "머리가 아프고 열이 나요"
}
```

**Query Parameter**
| 파라미터 | 타입 | 기본값 | 설명 |
|------|------|------|------|
| k | int | 3 | 반환할 Top-K 개수 |

**Response**
```json
[
  {"specialty": "내과", "score": 0.6521},
  {"specialty": "이비인후과", "score": 0.2134},
  {"specialty": "가정의학과", "score": 0.0891}
]
```

## 모델 상세

### 질병 예측 모델

| 항목 | 내용 |
|------|------|
| 베이스 모델 | `madatnlp/km-bert` (한국어 BERT) |
| 파인튜닝 체크포인트 | Hugging Face Hub (`cbnu-development-team-tuktak/disease-prediction-korean`) |
| 학습 | `BertForSequenceClassification` 파인튜닝 (10 Epoch) |
| 라벨 디코더 | sklearn `LabelEncoder` |

### 진료과 예측 모델

| 항목 | 내용 |
|------|------|
| 베이스 모델 | Hugging Face `AutoModelForSequenceClassification` |
| 체크포인트 | 로컬 저장 (`model/specialty_classifier`) |
| 라벨 디코더 | numpy `label_classes.npy` |

## 추론 파이프라인

```
사용자 텍스트 입력
      ↓
BertTokenizer / AutoTokenizer
(truncation=True, padding=True, max_length=512)
      ↓
BERT 모델 → logits 출력
      ↓
Softmax → 클래스별 확률 변환
      ↓
노이즈 클래스 마스킹 (-1.0 처리)
      ↓
torch.topk → Top-K 인덱스 및 확률 추출
      ↓
LabelEncoder / label_classes → 질병명/진료과명 디코딩
      ↓
[{"disease": "...", "score": 0.xxxx}, ...] 반환
```

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python app.py
# 포트 10000에서 실행됨
```
