# 포트폴리오 시뮬레이터

이 저장소는 Streamlit 앱 `app.py`를 포함합니다. Streamlit Cloud(share.streamlit.io) 또는 다른 호스팅 환경에 배포하려면 아래 단계를 따르세요.

배포 준비 체크리스트

- `requirements.txt`에 필요한 패키지 포함: `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`
- `.streamlit/config.toml` (이미 포함)로 서버 옵션 설정

빠른 배포 방법 (Streamlit Cloud)

1. GitHub에 이 레포를 푸시합니다.
2. https://share.streamlit.io 에 접속해 레포를 연결합니다.
3. 앱 경로에 `app.py`를 지정하고 배포합니다.

로컬에서 테스트

```bash
pip install -r requirements.txt
streamlit run app.py
```

팁

- 배포 후 외부 API(예: yfinance) 접근 문제가 있으면, 로그를 확인하고 `requirements.txt` 버전 고정을 고려하세요.
- 앱에 민감한 키가 있으면 `secrets`를 사용하세요.
