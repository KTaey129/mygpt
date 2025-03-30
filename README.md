# mygpt – using deepseek-r1

이 문서는 GCP VM에서 DeepSeek R1 모델을 활용하여 ChatGPT 유사 서비스를 구축하는 과정을 단계별로 설명합니다.

---

## 1. GCP 프로젝트 초기 설정

### 1.1. GCP 계정 생성 및 결제 설정
- **Google Cloud Console** 접속 후, Google 계정으로 로그인
- 결제(Billing) 프로필 등록하여 GCP 리소스 사용 설정

### 1.2. 프로젝트 생성
- GCP 콘솔 상단의 **프로젝트 선택 영역**에서 “프로젝트 만들기” 클릭
- 프로젝트 이름, 조직(또는 개인) 등 지정 후 생성

### 1.3. Billing 연결
- 생성한 프로젝트에 결제 계정 연결
- 여러 결제 계정이 있을 경우, 사용할 결제 계정 선택

### 1.4. 필수 API 활성화
- **Compute Engine**, **AI Platform(또는 Vertex AI)** 등 필요한 API 사용 설정
- Google Cloud Console → **APIs & Services → Library**에서 각 API 검색 후 Enable

---

## 2. IAM(권한) 및 서비스 계정 설정

### 2.1. IAM 정책 확인
- (조직이 있다면) 프로젝트 내 본인의 권한(Owner, Editor 등) 확인

### 2.2. 서비스 계정 생성
- 특정 작업(예: VM에서 Cloud Storage 접근, 로깅, 기타 API 호출)을 위한 서비스 계정 생성
- GCP 콘솔 → **IAM & Admin → Service Accounts → Create Service Account**
  - 예: `deepseek-service-account` 로 생성
- 필요한 역할(roles): 최소 권한(예: Compute Admin, Storage Admin)만 부여하여 보안 강화

### 2.3. 키(credential) 발급 (필요 시)
- VM 내부에서 gcloud CLI 인증이 필요하면 JSON 키 파일 다운로드
- 가능하면 **Compute Engine Default Service Account** 또는 직접 연결 방식 사용하여 JSON 키 파일 없이 진행

---

## 3. Compute Engine 인스턴스 생성 (GPU 사용 시)

### 3.1. 머신 타입 및 GPU 선택
1. **Compute Engine → VM Instances** 메뉴에서 “Create Instance” 클릭
2. 머신 구성:
   - **Machine family:** N1, N2, N2D, A2(GPU용) 등 선택 가능
   - **Machine type:** 적절한 vCPU/메모리 선택 (예: 소규모 테스트는 vCPU 4~8)
3. GPU 추가:
   - **GPUs:** NVIDIA Tesla T4, P100, V100, A100 등 선택 (예산과 추론 성능 고려)
   - GPU 수는 소규모 서비스라면 1개로 충분할 수 있음  
   *GPU 추가 시, 해당 존의 머신 타입 및 할당량(Quota) 확인 필요*

### 3.2. 부팅 디스크 및 OS
1. 부팅 디스크:
   - OS: Ubuntu 20.04 LTS 또는 22.04 LTS(또는 Debian) 권장
   - 디스크 용량: DeepSeek R1 모델 크기 + 추가 패키지/로그/데이터 고려 (예: 50GB ~ 100GB 이상)
2. 기타 옵션:
   - 방화벽 설정에서 HTTP(80) / HTTPS(443) 트래픽 허용 (웹 서버로 사용 시)
   - VPC, 서브넷, 영역(zone) 설정 (예: 서울 리전 → asia-northeast3)

### 3.3. 생성 후 SSH 접속
- 인스턴스 생성 완료 후 VM 목록에서 해당 인스턴스 선택 → “SSH” 버튼 클릭하여 웹 SSH 접속
- 또는 로컬에서:
  ```bash
  gcloud compute ssh <INSTANCE_NAME>

## 4. VM 내부 환경 설정

### 4.1. OS 업데이트 및 필수 패키지 설치
```bash
sudo apt-get update && sudo apt-get upgrade -y
# 필요 패키지 설치
sudo apt-get install -y build-essential git curl wget python3-pip
4.2. GPU 드라이버 및 CUDA/cuDNN 설치
NVIDIA 드라이버

GPU 사용 VM 생성 시, “Install GPU driver automatically” 옵션 체크 또는 수동 설치

수동 설치 예시:

bash
복사
sudo apt-get install -y nvidia-driver-###
CUDA 및 cuDNN

모델 요구사항(PyTorch, TensorFlow 버전에 맞는 CUDA 버전 설치

cuDNN도 동일 버전 계열로 설치

설치 후 정상 인식 여부 확인:

bash
복사
nvidia-smi
GPU 이름과 드라이버 버전 등이 출력되면 정상입니다.

4.3. Python 가상환경 구성
bash
복사
# Python 버전 확인 (Ubuntu 20.04/22.04 기본 3.8+ 내장)
python3 --version
# 가상환경 생성 (venv 예시)
python3 -m venv venv
source venv/bin/activate
# pip 업그레이드
pip install --upgrade pip
4.4. DeepSeek R1 모델 설치
모델 저장소(GitHub 등)에서 clone 받거나, PyPI/Conda 등으로 설치

예시:

bash
복사
git clone https://github.com/.../deepseek-r1.git
cd deepseek-r1
pip install -r requirements.txt
모델 체크포인트 파일(.pt, .bin 등)을 다운로드하여 VM 내 특정 경로에 저장:

bash
복사
wget https://example.com/deepseek_r1.ckpt -O /home/<USER>/models/deepseek_r1.ckpt
(실제 URL이나 파일 경로는 모델 제공처에 따라 달라질 수 있습니다.)

5. 웹 서버/Inference API 구성
5.1. FastAPI (예시)
5.1.1. FastAPI 설치
bash
복사
pip install fastapi uvicorn
5.1.2. 샘플 코드 작성 (예: app.py)
python
복사
# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
# from deepseek_r1 import DeepSeekR1  # 실제 모델 import 예시

app = FastAPI()

# 모델 로드 (샘플)
# model = DeepSeekR1.load_model("/home/ubuntu/models/deepseek_r1.ckpt")

class ChatRequest(BaseModel):
    user_query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    # response = model.inference(request.user_query)
    # 데모용 임시 응답
    response = f"Echo: {request.user_query}"
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
5.1.3. 서버 실행
bash
복사
python app.py
VM 외부에서는 <External IP>:8000/docs 또는 http://<External IP>:8000/chat를 통해 접근할 수 있습니다. (방화벽/네트워크 설정 확인 필요)

5.2. 방화벽/네트워크 점검
외부 IP에서 8000 포트에 접근이 안 될 경우:

GCP 콘솔 → VPC Network → Firewall rules 에서 포트 8000 허용 규칙 확인

“Allow HTTP/HTTPS traffic” 외 사용자 정의 포트(8000)는 별도 규칙 필요

80/443 포트로 서비스하려면 Nginx나 Caddy 등의 리버스 프록시 사용 가능

6. HTTPS 적용 및 도메인 연결 (선택 사항)
6.1. 도메인 구매 및 DNS 설정
이미 보유 중인 도메인이 있다면, DNS A 레코드를 GCP VM의 외부 IP로 연결

도메인이 없다면, Google Domains, 가비아, 후이즈 등에서 구매

6.2. Nginx 설정 예시
6.2.1. Nginx 설치
bash
복사
sudo apt-get install -y nginx
6.2.2. 리버스 프록시 설정
/etc/nginx/sites-available/default (또는 새 conf 파일)에 아래 설정 추가:

nginx
복사
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
6.2.3. HTTPS(SSL) 인증서 발급
Certbot을 사용하여 무료 Let’s Encrypt 인증서 발급:

bash
복사
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
인증서 발급 후, Nginx 설정이 자동으로 HTTPS로 업데이트 됩니다.

6.2.4. Nginx 재시작 및 상태 확인
bash
복사
sudo systemctl restart nginx
sudo systemctl status nginx
이후 브라우저에서 https://your-domain.com을 통해 FastAPI 서비스 동작 확인

7. 로깅/모니터링
7.1. Stackdriver Logging/Monitoring
GCP 콘솔의 Logging / Monitoring 메뉴에서 VM 로그, CPU/GPU 사용량, 메모리, 네트워크 트래픽 등을 모니터링

임계치(예: CPU 사용 80% 이상, GPU 메모리 부족 등) 설정 시 알림(Alerts) 구성 가능

7.2. 에러 로깅
FastAPI와 Python 로깅 기능을 활용하여 서버 에러를 Cloud Logging에 전송 가능
(예: logging 라이브러리로 stdout에 출력하면 GCP가 자동 수집)

8. 스냅샷 백업 및 유지 관리
8.1. 디스크 스냅샷
Compute Engine → Snapshots에서 VM 부팅/데이터 디스크의 스냅샷 생성

모델 파일이나 중요 설정의 주기적 백업을 통해 신속한 복원 가능

8.2. 자동 스냅샷 스케줄러
GCP에서 스냅샷 스케줄러 설정 (매일 혹은 매주 백업)

스냅샷은 지역/영역 단위로 저장되므로, DR(재해 복구) 목표에 맞게 여러 지역에 백업 고려

8.3. VM 업그레이드/다운그레이드
추론 속도 미흡 시, GPU를 더 강력한 모델(V100, A100)로 교체하거나 메모리 증설

사용량이 낮을 경우, 머신 타입을 낮춰 비용 절감 가능

9. 배포 자동화 (CI/CD) (선택 사항)
9.1. CI/CD 도구 활용
Cloud Build 또는 GitHub Actions를 활용하여 코드 빌드 및 GCP 자동 배포 구축

9.2. Docker 이미지
FastAPI + DeepSeek R1 모델을 도커라이징하여 VM, Cloud Run, GKE 등에서 실행

GCP Container Registry/Artifact Registry에 이미지를 push & pull 방식으로 배포

요약
프로젝트 생성 및 Billing 연결

IAM/Service Account 설정

Compute Engine 인스턴스 생성 (GPU 필요 시 GPU 옵션 선택)

VM 접속 후 환경 구성 (CUDA, cuDNN, Python, 모델 설치 등)

Inference API (예: FastAPI) 작성 및 포트 개방

HTTPS 적용 및 도메인 연결 (선택 사항)

로그/모니터링, 스냅샷 백업 설정

(선택) CI/CD 및 Docker를 활용한 자동화

이 과정을 통해 GCP VM에서 DeepSeek R1 모델을 로드하여 ChatGPT 유사 서비스를 실행할 수 있으며, 이후 모델 최적화, 리소스 조정, 보안 강화 등을 추가 진행할 수 있습니다.

추가적으로 궁금한 세부사항(예: 모델 최적화 방법, 비용 계산, 도커 설정 등)이 있으면 언제든지 문의해 주세요.

CPU 기반 DeepSeek R1 서비스 구현
1. 가상환경 및 의존성 설치
1.1. 가상환경 생성
bash
복사
python3 -m venv venv
source venv/bin/activate
1.2. 필요 라이브러리 설치
DeepSeek R1 모델의 requirements.txt 파일이 있는 경우:

bash
복사
pip install -r requirements.txt
CPU 기반 실행에 필요한 패키지(예: PyTorch CPU 버전, FastAPI, Uvicorn 등) 설치

2. 모델 로딩 및 추론 테스트
2.1. 테스트 스크립트 작성
python
복사
# test_inference.py
import torch
# from deepseek_r1 import DeepSeekR1  # 실제 모델 import 방식에 맞게 수정

# 모델 로드 (CPU 기반)
# model = DeepSeekR1.load_model("path/to/deepseek_r1.ckpt", map_location=torch.device('cpu'))

def test_inference():
    sample_input = "안녕하세요, 오늘 날씨는 어떤가요?"
    # output = model.inference(sample_input)
    output = "모델 테스트: " + sample_input  # 임시 응답
    print("추론 결과:", output)

if __name__ == "__main__":
    test_inference()
2.2. 실행 및 성능 확인
bash
복사
python test_inference.py
CPU 환경에서 추론 시간 및 메모리 사용량을 확인합니다.

3. FastAPI 기반 Inference API 구축
3.1. API 엔드포인트 구현 (main.py)
python
복사
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
# import torch
# from deepseek_r1 import DeepSeekR1

app = FastAPI()

# CPU 환경에서 모델 로드 (예시)
# model = DeepSeekR1.load_model("path/to/deepseek_r1.ckpt", map_location=torch.device('cpu'))

class ChatRequest(BaseModel):
    user_query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # response = model.inference(request.user_query)
    response = f"Echo (CPU): {request.user_query}"  # 임시 응답
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
3.2. API 서버 실행 및 테스트
bash
복사
uvicorn main:app --reload --host 0.0.0.0 --port=8000
브라우저, Postman 또는 curl 등을 이용하여 http://<인스턴스_IP>:8000/chat 호출 후 JSON 응답을 확인합니다.

4. 간단한 UI (프론트엔드) 구성 (선택 사항)
4.1. 정적 HTML 파일 작성 (static/index.html)
html
복사
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <title>DeepSeek R1 Chat (CPU)</title>
</head>
<body>
    <h1>DeepSeek R1 Chat</h1>
    <textarea id="userInput" rows="3" cols="50" placeholder="질문을 입력하세요..."></textarea>
    <br/>
    <button onclick="sendMessage()">전송</button>
    <div id="chatLog"></div>

    <script>
    async function sendMessage() {
        const query = document.getElementById('userInput').value;
        if (!query) return;
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_query: query })
        });
        const data = await response.json();
        const chatLog = document.getElementById('chatLog');
        chatLog.innerHTML += `<p><strong>You:</strong> ${query}</p>`;
        chatLog.innerHTML += `<p><strong>AI:</strong> ${data.answer}</p>`;
    }
    </script>
</body>
</html>
4.2. 정적 파일 서빙 설정 (FastAPI)
python
복사
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
브라우저에서 http://<인스턴스_IP>:8000/static/index.html에 접속하여 UI가 정상 작동하는지 확인합니다.
