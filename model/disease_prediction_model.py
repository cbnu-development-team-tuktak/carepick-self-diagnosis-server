# PyTorch 관련 import
import torch  # 텐서 및 모델 가중치 로딩용

# Transformers 관련 import
from transformers import BertTokenizer, BertForSequenceClassification  # 토크나이저 및 BERT 분류 모델

# Hugging Face Hub 관련 import
from huggingface_hub import hf_hub_download # 모델 체크포인트 원격 다운로드용

# 체크포인트 경로 설정
repo_id = "cbnu-development-team-tuktak/disease-prediction-korean"
filename = "disease_classifier_epoch10.pt"
model_name = "madatnlp/km-bert"

# 디바이스 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 학습 및 추론에 사용할 장치

# 체크포인트 로드
checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# 클래스 수 계산
num_labels = len(checkpoint["label_encoder"].classes_)  # 분류할 클래스 수

# 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained(model_name)  # 사전학습된 BERT 토크나이저 로딩
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  # BERT 분류 모델 생성
model.load_state_dict(checkpoint["model_state_dict"])  # 저장된 가중치 적용
model.to(device)  # 디바이스로 이동
model.eval()  # 평가 모드로 설정

# 라벨 인코더 로딩
label_encoder = checkpoint["label_encoder"]  # 라벨 → 인덱스 매핑 정보 포함

# 외부 모듈에서 import할 수 있도록 export
__all__ = ["model", "tokenizer", "label_encoder", "device"]