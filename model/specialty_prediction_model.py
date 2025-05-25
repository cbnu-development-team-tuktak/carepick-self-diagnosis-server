# PyTorch 관련 import
import torch  # 텐서 및 모델 가중치 로딩용

# Transformers 관련 import
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 토크나이저 및 BERT 분류 모델

# 모델 디렉토리 경로 설정
model_name = "model/specialty_classifier"  # 로컬 Hugging Face 포맷 모델 디렉토리 경로

# 디바이스 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 학습 및 추론에 사용할 장치

# 토크나이저 및 모델 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 저장된 토크나이저 로딩
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # 저장된 분류 모델 로딩
model.to(device)  # 디바이스로 이동
model.eval()  # 평가 모드로 설정

# 라벨 인코더 없음 → 라벨 클래스 수동 정의
label_classes = [
    '가정의학과', '감염내과', '내분비대사내과', '류마티스내과', '마취통증의학과', '비뇨의학과', '산부인과',
    '성형외과', '소아청소년과', '소화기내과', '순환기내과', '신경과', '신경외과', '신장내과', '안과',
    '영상의학과', '외과', '응급의학과', '이비인후과', '재활의학과', '정신건강의학과', '정형외과', '치과',
    '피부과', '혈액종양내과', '호흡기내과', '흉부외과'
]

# 외부 모듈에서 import할 수 있도록 export
__all__ = ["model", "tokenizer", "label_classes", "device"]