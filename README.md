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

## 🔧 문제 및 해결

### ❓ 모델 파일의 GitHub 업로드 용량 제한 문제 
#### 상황
10 epoch 이상 학습한 BERT 기반 질병 예측 모델은 파일 용량은 GitHub의 업로드 제한(100MB)을 초과하여 버전 관리가 불가능하였습니다 
#### 해결
- **Git LFS (Large File Storage)**을 도입하여 모델 파일을 GitHub에 안정적으로 업로드하고 버전 관리할 수 있도록 처리하였습니다. 
- Git의 확장 기능에 대한 이해도를 높였고, 팀원들과 대용량 모델 파일을 안정적으로 공유할 수 있는 기반을 마련했습니다.

---
  
### ❓ 모델 학습 전 라우트 테스트를 위한 Dummy 모델 구성
#### 상황
진료과 예측 모델이 학습되기 전, 전체 백엔드 흐름을 테스트하기 어려웠고, 팀 개발 속도에 영향을 줄 수 있었습니다. 
#### 해결
- 실제 모델 구조와 동일한 형태로 작동하는 **Dummy 모델**(또는 Test 모델)을 구성하여, 라우트를 선제적으로 구현하고 테스트를 진행했습니다.
- 추후 학습된 모델로의 대체가 원활했고, 병목 없이 API 연동과 프론트엔드 작업이 동시에 진행될 수 있었습니다.

---

### ❓ Render.com 배포 실패 이슈
#### 상황
EC2에서 VS Code Remote SSH로의 접속이 지속적으로 실패하여, 대안으로 Render.com을 통한 배포를 시도하였습니다. <br>
그러나 배포 과정에서 모델 로딩 시점마다 메모리 초과로 인해 서버가 자동을 종료되며, 서비스가 정상적으로 배포되지 않는 문제가 반복되었습니다.
#### 해결
- Render.com Starter플랜의 메모리 한도(512MB)가 BERT 기반 모델 초기 로딩에 부족함을 확인했습니다.
- 결국 Render에서는 해당 모델을 안정적으로 배포하는 것이 불가능하다고 판단하여, **EC2 서버로 직접 배포 및 리소스 확장 환경을 구성**하는 방향으로 전환하였습니다.

---

### ❓ EC2 인스턴스 접속 실패 이슈
#### 상황
**EC2 인스턴스를 시작한 직후**에는 VSCode Server를 통한 SSH 접속이 정상적으로 작동했으나, <br>
일정 시간이 지난 후에는 연결이 갑자기 끊기고, 이후로는 **타임아웃 오류로 재접속이 불가능**했습니다. <br>
마찬가지로 WinSCP 접속 시도에서도 같은 증상이 반복되었습니다.
#### 해결
- EC2의 **기본 네트워크 구성 요소**를 점검 및 수정하여 정상 작동을 확인하였습니다.
  - **서브넷**: 퍼블릭 서브넷 여부 확인 및 가용 영역 설정 점검
  - **인터넷 게이트웨이 (IGW)**: 해당 VPC에 IGW가 **연결되어 있는지** 확인
  - **라우팅 테이블**: IGW로 향하는 **0.0.0.0/0** 라우팅 항목이 등록되어 있는지 확인
- **보안 그룹** 설정을 확인하고 인바운드 규칙을 추가하였습니다.
  - **애플리케이션 포트**, TCP 10000번을 등록하였습니다.
- VSCode Server 방식은 안정성이 떨어진다고 판단해 중단하였습니다.
- 최종적으로 WinSCP를 별도 설정을 통해 연결하고, 서버 소스 코드를 복제하는 데 성공하였습니다.

---

### ❓ EC2 인스턴스 내 패키지 설치 중 용량 부족 문제 및 모델 로드 시 메모리 부족 문제
#### 상황
서버 소스 코드를 성공적으로 복제한 후, `pip install -r requirements.txt` 명령어로 패키지를 설치하려고 했으나 **디스크 용량 부족 오류**로 인한 실패하였습니다. <br>
추가적으로, BERT 기반 모델 실행으로 **메모리 부족으로 인한 프로세스 강제 종료(Killed)**의 가능성도 고려했어야 했습니다.
#### 해결
- EC2 인스턴스의 **EBS 볼륨 크기**를 기존 8GB에서 **16GB로 확장**하였습니다.
  - AWS 콘솔에서 EBS → 볼륨 → 크기 수정 후, EC2 내에서 `growpart`, `resize2fs` 명령어로 파일시스템까지 확장하였습니다.
- BERT 모델 구동을 위한 **가상 메모리(Swap)** 공간을 추가로 할당하였습니다.
  - `sudo fallocate -l 4G /swapfile` 등을 통해 **4GB 스왑 메모리**를 구성했습니다.
  - `/etc/fstab`에 등록하여 EC2 재부팅 시에도 스왑이 유지되도록 설정하였습니다.
- 이를 통해 모델 로딩과 추론이 안정적으로 수행되었으며, 프리 티어 환경에서도 **중단 없는 실행 및 API 응답 제공이 가능**해졌습니다.
