# model/__init__.py

# 질병 분류 모델 관련 객체 import
from model.disease_prediction_model import model as disease_model # 질병 분류 모델
from model.disease_prediction_model import tokenizer as disease_tokenizer # 질병 분류용 BERT 토크나이저
from model.disease_prediction_model import label_encoder as disease_label_encoder # 질병 라벨 인코더 (LabelEncoder)
from model.disease_prediction_model import device as disease_device # 질병 분류 모델 실행 디바이스 (GPU/CPU)

# 진료과 분류 모델 관련 객체 import
from model.specialty_prediction_model import model as specialty_model # 진료과 분류 모델
from model.specialty_prediction_model import tokenizer as specialty_tokenizer # 진료과 분류용 BERT 토크나이저  
from model.specialty_prediction_model import label_classes as specialty_label_classes # 진료과 라벨 리스트
from model.specialty_prediction_model import device as specialty_device # 진료과 분류 모델 실행 디바이스 (GPU/CPU)

__all__ = [
    "disease_model", "disease_tokenizer", "disease_label_encoder", "disease_device", # 질병 관련
    "specialty_model", "specialty_tokenizer", "specialty_label_classes", "specialty_device" # 진료과 관련
]
