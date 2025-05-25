# Transformer 관련 import
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification # 토크나이저, 설정, BERT 분류 모델 관련 모듈
# OS 관련 import
import os # 디렉토리 생성 등 파일 시스템 작업용

model_name = "madatnlp/km-bert" # 사용할 시간학습 BERT 모델명 
save_path = "model/specialty_classifier" # 저장할 로컬 디렉토리 경로

os.makedirs(save_path, exist_ok=True) # 디렉토리가 없을 경우 생성

config = BertConfig.from_pretrained(model_name, num_labels=27) # 분류 클래스 수를 반영한 BERT 설정 객체 생성

# 설정 기반으로 분류용 BERT 모델 초기화
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# 해당 모델에 맞는 토크나이저 로딩
tokenizer = BertTokenizer.from_pretrained(model_name)

# 모델 가중치와 설정 파일을 지정된 경로에 저장
model.save_pretrained(save_path)

# 토크나이저 관련 파일(vocab 등)을 동일 경로에 저장
tokenizer.save_pretrained(save_path)

# 저장 완료 로그 출력
print(f"[INFO] dummy 모델이 '{save_path}'에 저장되었습니다.")
