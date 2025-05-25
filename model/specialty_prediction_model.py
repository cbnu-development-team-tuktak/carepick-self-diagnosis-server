# PyTorch 관련 import
import torch  # 텐서 및 모델 가중치 로딩용
import numpy as np

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

label_classes = np.load(f"{model_name}/label_classes.npy", allow_pickle=True).tolist()

# 외부 모듈에서 import할 수 있도록 export
__all__ = ["model", "tokenizer", "label_classes", "device"]