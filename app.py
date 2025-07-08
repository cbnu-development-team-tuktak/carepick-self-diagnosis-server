# Flask 관련 import
from flask import Flask # Flask 앱 생성용 import

# 라우트 관련 import
from routes.disease import disease_bp # 질병 예측 라우트 import
# from routes.specialty import specialty_bp # 진료과 예측 라우트 import 
from routes.specialty import specialty_bp

def create_app():
    app = Flask(__name__) # Flask 애플리케이션 인스턴스 생성
    app.register_blueprint(disease_bp, url_prefix="/disease") # 질병 예측 라우트 등록
    app.register_blueprint(specialty_bp, url_prefix="/specialty") # 진료과 예측 라우트 등록
    return app # 앱 객체 반환

# Flask 앱 실행 엔트리포인트
if __name__ == "__main__":
    app = create_app() # Flask 앱 생성
    app.run(host="0.0.0.0", port=10000) # 모든 IP에서 접근 가능하도록 서버 실행 (포트 10000)
