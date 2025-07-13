
# 🔍 AI_Model_2 배치 테스트 시스템 (선용품 표준코드 자동 매칭)

선용품 데이터셋 기반으로 품목명을 자동 전처리하고, 계층 분류 및 유사도 기반 검색을 통해 표준 코드를 추천하는 배치 테스트용 노트북입니다.

---

## 📁 주요 목적

- 품목명 일괄 입력 후 자동 분류 예측 (L1 ~ L4)
- 유사 품목 검색 및 P CODE 추천
- 유효 조합 필터링
- 신규 품목 대응 로직 포함

---

## 🧩 사용 구성 요소 (필수 모델)

아래 모든 파일이 `AI_Model_2/` 폴더에 존재해야 합니다:

| 파일명 | 설명 |
|--------|------|
| `L1_NAME_model.pkl` | L1 계층 분류기 |
| `L2_NAME_model.pkl` | L2 계층 분류기 |
| `L3_NAME_model.pkl` | L3 계층 분류기 |
| `L4_NAME_model.pkl` | L4 계층 분류기 |
| `L1_NAME_vectorizer.pkl` | L1 벡터라이저 |
| `L2_NAME_vectorizer.pkl` | L2 벡터라이저 |
| `L3_NAME_vectorizer.pkl` | L3 벡터라이저 |
| `L4_NAME_vectorizer.pkl` | L4 벡터라이저 |
| `L1_NAME_label_encoder.pkl` | L1 라벨 인코더 |
| `L2_NAME_label_encoder.pkl` | L2 라벨 인코더 |
| `L3_NAME_label_encoder.pkl` | L3 라벨 인코더 |
| `L4_NAME_label_encoder.pkl` | L4 라벨 인코더 |
| `searchname_vectorizer.pkl` | TF-IDF 벡터라이저 (품목명 검색용) |
| `preprocessor.pkl` | 전처기를 거친 데이터 파일|
| `df_model.pkl` | 표준 품목 전체 데이터셋 |
| `valid_combinations.pkl` | 유효 조합 목록 |

---

## ⚙️ 설치

```bash
pip install pandas numpy scikit-learn lightgbm joblib openpyxl
```

---

## 📂 구조

```
project/
├── AI_Mode_2_edited.ipynb         # 배치 테스트용 노트북
├── AI_Model_2/                    # 모든 모델과 벡터/인코더 파일
│   ├── L1_NAME_model.pkl
│   ├── L1_NAME_vectorizer.pkl
│   ├── L1_NAME_label_encoder.pkl
│   ├── ...
│   ├── searchname_vectorizer.pkl
│   ├── df_model.pkl
│   ├── preprocessor.pkl
│   └── valid_combinations.pkl
└── input/
    └── 선용품 표준코드 목록 - ALL.xlsx
```
---

## 실행 방법 (모델 학습도 함꼐 진행)

1. Jupyter에서 `AI_Mode_2_edited.ipynb` 열기
2. 상단부터 셀 순차 실행
3. 결과 출력 자동 진행

## 모델만 불러와서 실행하는 방법


---

## 🔧 주요 처리 흐름

### 1. 전처리기 정의 및 학습
- 불용어 제거, 단위 인식
- 텍스트 정규화 및 대체 수행

### 2. 계층별 분류 예측
```python
for col in ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']:
    model = joblib.load(...)
    vectorizer = joblib.load(...)
    label_encoder = joblib.load(...)
```
```
import joblib
import os

# 모델 파일들이 저장된 경로
load_dir = "AI_Model_2"

# 계층 컬럼명 리스트
required_cols = ['L1 NAME', 'L2 NAME', 'L3 NAME', 'L4 NAME']

# 모델 및 벡터라이저 저장 딕셔너리
hierarchical_models = {}
hierarchical_label_encoders = {}

# 각 계층에 대해 필요한 모델들 로딩
for col in required_cols:
    # 모델 불러오기
    model_path = os.path.join(load_dir, f"{col}_model.pkl")
    model = joblib.load(model_path)

    # 벡터라이저 불러오기
    vectorizer_path = os.path.join(load_dir, f"{col}_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)

    # 라벨 인코더 불러오기
    encoder_path = os.path.join(load_dir, f"{col}_label_encoder.pkl")
    encoder = joblib.load(encoder_path)

    # 저장
    hierarchical_models[col] = (model, vectorizer)
    hierarchical_label_encoders[col] = encoder

print("모든 계층의 모델, 벡터라이저, 라벨 인코더 로딩 완료!")
```
### 3. 유효 조합 확인
- 예측된 L1~L4가 `valid_combinations.pkl` 내 존재하는지 검증

### 4. 유사 품목 추천
- TF-IDF 기반 코사인 유사도 계산
- Top-N 품목 추천 및 P CODE 반환

---

## ✅ 결과 출력 예시

```text
🔎 입력: DECK EQUIPMENT
→ 전처리 후: DECK EQUIPMENT

📂 예측: L1 = DECK, L2 = EQUIPMENT, L3 = WINCH, L4 = MOORING WINCH
✔️ 조합 유효

📌 유사 품목 추천:
   - MOORING WINCH (P CODE: DK021134) [유사도: 0.812]
   - HYDRAULIC WINCH (P CODE: DK019002) [유사도: 0.789]
```
실제로 실행하면 더 세부적인 단계로 나누어서 출력이 진행됩니다!

---

## 🧠 신뢰도 기준

| 유사도 | 의미 | 처리 방안 |
|--------|------|------------|
| ≥ 0.80 | 매우 정확 | 자동 등록 가능 |
| 0.6~0.8 | 일반적 정확도 | 검토 후 사용 |
| 0.4~0.6 | 보통 | 신규 코드 생성 고려 |
| < 0.4 | 낮음 | 검토 또는 수동 분류 |

---

## ✨ 장점 및 확장성

- 텍스트 기반 품목 입력만으로 전체 분류 예측
- 신규 품목 대응 가능
- 동의어 및 유사품 자동 보정
- 대규모 입력도 배치 처리 가능

---

## 🗓 버전 및 이력

- **버전**: 1.0.0
- **작성일**: 2025년 7월 12일
- **기반 코드**: `AI_Mode_2_edited.ipynb`
