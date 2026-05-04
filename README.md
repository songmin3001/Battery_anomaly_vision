# battery_anomaly_vision

배터리 외관 이상 자동 판별 시스템
EfficientNet-B3 + Grad-CAM + TTA 기반 불량 분류 및 이상 정도 정량화

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 모델 | EfficientNet-B3 (ImageNet 사전학습) |
| 클래스 | 정상 / 부풀림 / 찌그러짐 / 전해액유출 / 폭발 |
| 데이터 | 총 900장 (정상 300 + 불량 150×4) |
| 핵심 기법 | TTA (8가지 변환 평균), Grad-CAM 히트맵, 이상 스코어 |

---

## 폴더 구조

```
battery-visual-inspection/
├── data/
│   ├── normal/          # 300장
│   ├── swelling/        # 150장 (부풀림)
│   ├── dent/            # 150장 (찌그러짐)
│   ├── leakage/         # 150장 (전해액 유출)
│   └── explosion/       # 150장 (폭발)
├── src/
│   ├── dataset.py       # 데이터로더 + TTA 변환 정의
│   ├── model.py         # EfficientNet-B3 분류기
│   ├── train.py         # 학습 루프
│   ├── evaluate.py      # 평가 + 이상 스코어 + Grad-CAM
│   └── gradcam.py       # Grad-CAM 시각화 모듈
├── runs/                # 학습 결과 자동 저장
├── requirements.txt
└── README.md
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 사용법

### 1. 학습

```bash
cd src
python train.py \
  --data_dir ../data \
  --save_dir ../runs \
  --epochs 60 \
  --batch_size 16 \
  --lr 3e-4 \
  --patience 12
```

학습 완료 후 `../runs/best_model.pth` 와 `../runs/history.json` 이 저장됩니다.

### 2. 평가 (TTA 적용)

```bash
python evaluate.py \
  --model_path ../runs/best_model.pth \
  --data_dir ../data \
  --save_dir ../runs/eval \
  --gradcam
```

`--gradcam` 플래그를 붙이면 오분류 샘플에 대해 Grad-CAM 히트맵이 자동 생성됩니다.

### 3. 단일 이미지 Grad-CAM

```bash
python gradcam.py \
  --image path/to/battery.jpg \
  --model_path ../runs/best_model.pth \
  --save result_gradcam.jpg
```

---

## 주요 설계 포인트

### 이상 스코어 (Anomaly Score)
- 정상 클래스 소프트맥스 확률의 역수로 정의
- `anomaly_score = 1 - P(normal)`
- 0에 가까울수록 정상, 1에 가까울수록 심각한 불량

### TTA (Test-Time Augmentation)
원본 + 7가지 기하학적/광도 변환에 대해 각각 추론 후 확률 평균  
→ 단일 추론 대비 안정적인 경계면 인식 성능

### 클래스 불균형 보정
- 정상(300) vs 불량(150×4) 비율 차이를 `CrossEntropyLoss weight` 로 보정
- 불량 클래스에 자동으로 더 높은 가중치 부여

### Grad-CAM
- EfficientNet-B3 마지막 Conv 블록에 hook 등록
- 모델이 테두리 형상에 집중하는지 시각적으로 확인 가능

---

## 출력 예시

```
[예측] swelling
[클래스별 확률]
  normal       0.0312  ████
  swelling     0.8741  ████████████████████████████
  dent         0.0521  ██
  leakage      0.0298  ██
  explosion    0.0128  █

[이상 스코어] 0.9688  (0=정상, 1=심각한 불량)
```
