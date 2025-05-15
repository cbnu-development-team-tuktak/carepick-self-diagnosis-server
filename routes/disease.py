# Flask 관련 import
from flask import Blueprint, request, jsonify # 웹 요청 처리용 Flask 모듈

# 모델 로딩 관련 import
from model.loader import model, tokenizer, label_encoder, device # 사전학습된 모델, 토크나이저, 인코더, 디바이스 정보 불러오기

# 예측 함수 import
from inference.disease_predictor import predict_topk  # top-k 예측 수행 함수 불러오기

# Blueprint 설정 관련 코드
disease_bp = Blueprint("disease", __name__) # Flask Blueprint 객체 생성

# 요청에서 텍스트와 k값 추출
def extract_text_and_k():
    data = request.get_json()  # JSON 본문 수신
    text = data.get("text") if isinstance(data, dict) else None  # "text" 필드 추출
    text = text if isinstance(text, str) else None  # 문자열 여부 검증

    k = request.args.get("k", default=3, type=int)  # 쿼리 파라미터에서 k 추출, 기본값 3

    return text, k  # 둘 다 반환

# /disease/mini 라우트 정의
@disease_bp.route("/mini", methods=["POST"]) # POST 방식으로 /disease/mini 엔드포인트 등록
def predict_disease_mini():
    text, k = extract_text_and_k()  # 사용자 입력 텍스트와 k 동시 추출
    if not text:  # 텍스트가 없거나 문자열이 아니면
        return jsonify({"error": "No valid input text provided"}), 400  # 에러 메시지 반환
    
    # 예측 수행
    result, status = predict_topk(
        text, # 사용자 입력 텍스트
        tokenizer, # BERT 토크나이저
        model, # 학습된 분류 모델
        label_encoder, # 인덱스를 질병명으로 디코딩하는 라벨 인코더
        device, # 모델 실행 디바이스 (GPU 또는 CPU
        k # Top-k 예측 개수
    )  
    
    # Flask 라우트에서 반환할 예측 결과 및 상태 코드 출력
    print(f"[DEBUG] Flask 라우트 반환 결과: {jsonify(result)}, 상태 코드: {status}")
    return jsonify(result), status  # 예측 결과 JSON 형식으로 반환
    
# /disease/advanced 라우트 정의
@disease_bp.route("/advanced", methods=["POST"]) # POST 방식으로 /disease/advanced 엔드포인트 등록
def predict_disease_advanced():
    text, k = extract_text_and_k()  # 사용자 입력 텍스트와 k 동시 추출
    if not text:  # 텍스트가 없거나 문자열이 아니면
        return jsonify({"error": "No valid input text provided"}), 400  # 에러 메시지 반환
    # result, status = predict_topk(text, tokenizer, model, label_encoder, device, k)  # 예측 수행 (향후 실제 구현 예정)
    result = {"top5": []}  # 고정된 빈 결과 반환
    status = 404  # 아직 미구현된 기능으로 404 반환
    return jsonify(result), status  # 예측 결과 JSON 형식으로 반환

