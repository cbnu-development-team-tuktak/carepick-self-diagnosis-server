import torch
from torch.nn.functional import softmax

def predict_topk(
    text, # 사용자 입력 텍스트 (증상 또는 질의 설명)
    tokenizer, # 사전학습된 BERT 토크나이저
    model, # 학습된 BERT 분류 모델
    label_classes, # 진료과 레이블 리스트 (인덱스 → 레이블 변환용)
    device, # 추론에 사용할 디바이스 (GPU 또는 CPU)
    k=3 # 반환할 Top-k 진료과 수 
):
    # 입력된 문장을 로그로 출력
    print(f"[INFO] 예측 요청 수신 - 입력 텍스트: {text}")

    # 텍스트가 비어있거나 문자열이 아닌 경우
    if not text or not isinstance(text, str):
        # 에러 로그 출력
        print("[ERROR] 입력 텍스트가 유효하지 않음")
        
        # 에러 메시지와 400 코드 반환
        return {"error": "No valid input text provided"}, 400

    # 토크나이징 시작 로그 출력
    print("[INFO] 토크나이징 시작")
    
    # 입력 문장 토크나이징 및 텐서 변환
    encoding = tokenizer(
        text, # 입력 문장
        return_tensors="pt", # PyTorch 텐서 형태로 변환
        truncation=True, # 최대 길이를 초과하는 경우 잘라냄
        padding=True, # 길이를 맞추기 위해 패딩 추가
        max_length=512 # 최대 토큰 길이 설정
    )

    # 토크나이징 완료 로그 출력
    print("[INFO] 토크나이징 완료")

    # 입력 토큰 ID 시퀀스를 디바이스로 이동 (GPU 또는 CPU)
    input_ids = encoding["input_ids"].to(device)
    
    # 패딩 여부를 나타내는 마스크도 디바이스로 이동
    attention_mask = encoding["attention_mask"].to(device)

    # 예측 시작 안내 로그 출력
    print("[INFO] 모델 추론 시작")
    
    # 예측 수행 (추론 모드)
    with torch.no_grad(): # 그래디언트 계산 비활성화
        # 모델에 입력값 전달하여 로짓(logits) 출력
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 소프트맥스를 통해 클래스별 확률로 변환
        probs = softmax(output.logits, dim=1)
        
        # 확률이 높은 상위 k개 클래스의 인덱스와 확률 추출
        topk = torch.topk(probs, k=k, dim=1)

    # 모델 예측 완료 로그 출력
    print("[INFO] 모델 추론 완료")

    # top-k 클래스 인덱스를 리스트로 변환
    indices = topk.indices[0].tolist()
    
    # top-k 클래스의 예측 확률을 리스트로 변환
    scores = topk.values[0].tolist()

    # softmax 결과 중 확률이 높은 상위 k개 클래스의 인덱스를 출력
    print(f"[INFO] Top-{k} 인덱스: {indices}")
    
    # 해당 인덱스에 대응하는 예측 확률 값을 출력
    print(f"[INFO] Top-{k} 확률: {scores}")

    try:
        # 예측된 인덱스를 실제 진료과명(클래스 라벨)으로 디코딩
        specialties = [label_classes[i] for i in indices]
    except Exception as e: # 디코딩 중 오류가 발생한 경우
        print(f"[ERROR] 라벨 디코딩 실패: {str(e)}") # 에러 로그 출력
    
        # 에러 메시지와 500 코드 반환
        return {"error": f"Label decoding error: {str(e)}"}, 500 
    
    # 예측 결과 구성
    result = [
        {"specialty": s, "score": round(score, 4)} # 진료과명과 예측 확률(소수점 4자리)을 딕셔너리 형태로 저장
        for s, score in zip(specialties, scores) # 진료과명과 확률을 페어로 묶어서 순회
    ]

    # 최종 예측 결과 로그 출력
    print(f"[INFO] 최종 예측 결과: {result}")
    
    # 예측 결과와 성공 상태 코드 반환
    return result, 200
