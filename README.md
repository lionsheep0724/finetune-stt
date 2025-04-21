# Whisper v3 Turbo 한국어 STT 파인튜닝

이 프로젝트는 Whisper v3 Turbo 모델을 한국어 음성 인식(STT)에 맞게 파인튜닝하기 위한 코드를 제공합니다.

## 요구 사항

필요한 패키지를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 데이터 형식

데이터는 다음과 같은 형식으로 구성되어야 합니다:
- `.wav` 파일: 오디오 데이터
- `.txt` 파일: 해당 오디오의 텍스트 트랜스크립션 (동일한 파일명)

예시:
```
data/
  ├── sample1.wav
  ├── sample1.txt
  ├── sample2.wav
  ├── sample2.txt
  └── ...
```

## 학습 방법

학습을 시작하려면 다음 명령어를 실행하세요:

```bash
python train.py --data_dir ./data --output_dir ./whisper-v3-turbo-ko
```

### 주요 매개변수

- `--data_dir`: 데이터셋 디렉토리 경로 (기본값: `./data`)
- `--output_dir`: 훈련된 모델 저장 경로 (기본값: `./whisper-v3-turbo-ko`)
- `--batch_size`: 훈련 배치 크기 (기본값: 8)
- `--learning_rate`: 학습률 (기본값: 1e-5)
- `--max_steps`: 최대 학습 스텝 수 (기본값: 5000)
- `--fp16`: FP16 정밀도 사용 (기본값: False)
- `--chunk_length`: 오디오 청크 길이(초) (기본값: 30초)
  * 짧은 오디오 파일(2-3초)이 많은 경우 이 값을 5-10초로 줄이면 성능이 향상될 수 있습니다.
  * 예: `--chunk_length 5`
- `--logging_steps`: 로깅 간격 (기본값: 25)
- `--tensorboard_dir`: TensorBoard 로그 디렉토리 (기본값: output_dir과 동일)

모든 매개변수 목록을 보려면 다음 명령어를 실행하세요:
```bash
python train.py --help
```

## TensorBoard 모니터링

학습 과정은 TensorBoard를 통해 모니터링할 수 있습니다. Docker Compose를 사용하여 TensorBoard를 실행하세요:

```bash
docker-compose up -d
```

그런 다음 웹 브라우저에서 다음 주소로 접속하세요:
```
http://localhost:6006
```

주요 모니터링 지표:
- Loss: 학습 및 검증 손실
- WER (Word Error Rate): 단어 오류율
- CER (Character Error Rate): 문자 오류율

TensorBoard에서는 다음과 같은 탭을 통해 학습 과정을 모니터링할 수 있습니다:
- Scalars: 손실 및 메트릭 변화
- Graphs: 모델 구조
- Distributions: 가중치 분포
- Histograms: 가중치 히스토그램

TensorBoard는 학습이 진행되는 동안 실시간으로 업데이트됩니다.