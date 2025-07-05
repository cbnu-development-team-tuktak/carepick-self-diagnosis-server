# Carepick Self Diagnosis Server - 질병/진료과 예측 모델 서버

## 📄 프로젝트 소개
Carepick Self Diagnosis Server는 사용자의 증상 입력에 따라 사전 학습된 모델을 기반으로, <br>
질병 또는 진료과를 예측하여 Top-k 결과를 반환하는 Flask 기반 서버입니다. <br>

이 서버는 메인 백엔드(Spring Boot)와 분리되어 있어, <br>
**서버 부하를 분산**하고 **Python 기반의 사전 학습 모델 처리 효율성**을 극대화하기 위해 <br>
독립적으로 구성되어 있습니다.

## 🛠️ 가술 스택
