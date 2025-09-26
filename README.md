# 🩺 심장병 예측 모델링 (회귀 및 분류)

![Python](https://img.shields.io/badge/python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-2.x-blue?logo=pandas)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-blue?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/seaborn-0.x-blue?logo=seaborn)
![XGBoost](https://img.shields.io/badge/XGBoost-purple?logo=xgboost)

---

## 1. 프로젝트 개요

심장 질환은 전 세계적으로 주요 사망 원인 중 하나로, 조기 발견과 예측이 매우 중요합니다. 이 프로젝트에서는 환자의 의료 기록 데이터를 활용하여 **심장병 발병 여부를 예측하는 분류 모델**과 **최대 심박수를 예측하는 회귀 모델**을 개발하고, 각 모델의 성능을 비교 분석합니다.

### 분석 배경
- 데이터 기반의 질병 예측 모델은 의료진의 진단을 보조하고, 환자에게 조기 경고를 제공하여 예방적 치료의 기회를 높일 수 있습니다.
- 다양한 머신러닝 모델을 동일한 데이터에 적용하고 성능을 비교함으로써, 특정 문제에 가장 적합한 알고리즘을 선택하는 능력을 기릅니다.

### 분석 목표
- **분류(Classification):** 환자의 의료 기록을 바탕으로 심장병 발병 여부(`target`)를 예측하는 최적의 분류 모델을 찾는다.
- **회귀(Regression):** 환자의 나이, 혈압 등 다른 특성들을 바탕으로 최대 심박수(`thalach`)를 예측하는 최적의 회귀 모델을 찾는다.

### 사용 데이터
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

### 데이터 변수 설명 (Data Dictionary)

| 변수명 (Variable) | 설명 |
| :--- | :--- |
| **age** | 나이 |
| **sex** | 성별 (1 = male; 0 = female) |
| **cp** | 가슴 통증 유형 (Chest Pain Type) <br> (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic) |
| **trestbps** | 안정 시 혈압 (Resting Blood Pressure) (mm Hg) |
| **chol** | 혈청 콜레스테롤 (Serum Cholestoral) (mg/dl) |
| **fbs** | 공복 혈당 > 120 mg/dl (Fasting Blood Sugar) <br> (1 = true; 0 = false) |
| **restecg** | 안정 시 심전도 결과 (Resting Electrocardiographic Results) <br> (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy) |
| **thalach** | 최대 심박수 (Maximum Heart Rate Achieved) |
| **exang** | 운동 유발 협심증 (Exercise Induced Angina) <br> (1 = yes; 0 = no) |
| **oldpeak** | 안정 상태 대비 운동으로 유발된 ST 분절 하강 (ST depression) |
| **slope** | 최대 운동 ST 분절의 기울기 (Slope of the peak exercise ST segment) |
| **ca** | 형광 투시법으로 관찰된 주요 혈관의 수 (0-3) |
| **thal** | 지중해성 빈혈 여부 (Thalassemia) <br> (1: Normal, 2: Fixed defect, 3: Reversable defect) |
| **target** | **심장병 유무 (진단 결과)** <br> (0 = 없음, 1 = 있음) |

### 분석 도구
- `Python`, `Jupyter Notebook`
- `Pandas`, `NumPy`
- `Matplotlib`, `Seaborn`
- `Scikit-learn`

---

## 2. 분석 절차
1.  **데이터 불러오기 및 전처리:** 데이터의 기본 정보를 확인하고, 분석에 적합하도록 데이터를 정제합니다.
2.  **탐색적 데이터 분석 (EDA):** 각 변수별 분포와 변수 간의 관계를 시각화하여 데이터의 특징과 패턴을 파악합니다.
3.  **피처 엔지니어링:** 모델 학습에 더 효과적인 변수를 생성하거나 범주형 변수를 인코딩합니다.
4.  **분류 모델링:**
    - 로지스틱 회귀, 랜덤 포레스트, XGBoost 등 다양한 분류 모델을 학습합니다.
    - 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score, ROC AUC Score를 사용하여 모델 성능을 평가하고 비교합니다.
5.  **회귀 모델링:**
    - 선형 회귀, 릿지(Ridge), 라쏘(Lasso), 랜덤 포레스트 등 다양한 회귀 모델을 학습합니다.
    - RMSE(Root Mean Squared Error), R²(결정 계수)를 사용하여 모델 성능을 평가하고 비교합니다.
6.  **결론 및 모델 해석:** 최적 모델을 선정하고, 모델이 어떤 변수를 중요하게 판단했는지(Feature Importance) 해석합니다.

---

## 3. 주요 분석 결과 및 모델 성능

### 분류 모델 성능 비교

| 모델 (Model) | 정확도 (Accuracy) | F1-Score (for Target=1) |
| :--- | :---: | :---: |
| 로지스틱 회귀 | 0.82 | 0.84 |
| **🏆 랜덤 포레스트** | **0.83** | **0.85** |
| XGBoost | 0.82 | 0.84 |

**최적 모델:** **랜덤 포레스트**가 약 83%의 정확도와 0.85의 F1-Score로 가장 안정적이고 우수한 성능을 보였습니다.


### 회귀 모델 성능 비교

| 모델 (Model) | RMSE | R² (결정 계수) |
| :--- | :---: | :---: |
| 선형 회귀 | 21.53 | 0.235 |
| Ridge | 21.52 | 0.236 |
| Lasso | 21.61 | 0.229 |
| **🏆 랜덤 포레스트** | **21.26** | **0.254** |
| XGBoost | 22.79 | 0.143 |

**최적 모델:** **랜덤 포레스트**가 RMSE 21.26, R² 0.254로 가장 나은 성능을 보였습니다.

## 4. 결론

-   **분류 모델:** 랜덤 포레스트를 활용하여 **약 83%의 정확도로 심장병 발병 여부를 예측**하는 모델을 구축했습니다. 이는 의료진의 조기 진단을 보조하는 도구로서의 가능성을 보여줍니다.
-   **회귀 모델:** R² 값이 약 0.25로 낮게 나타나, **현재 데이터만으로는 환자의 최대 심박수를 정확히 예측하는 데 한계가 있음**을 확인했습니다. 이는 모델의 실패가 아닌, 데이터의 내재적 한계를 발견한 유의미한 분석 결과입니다.
-   **향후 개선 방안:** `GridSearchCV`를 이용한 하이퍼파라미터 튜닝, 추가적인 피처 엔지니어링, 또는 환자의 생활 습관과 같은 추가 데이터 확보를 통해 모델 성능을 더 향상시킬 수 있을 것입니다.


---

## 5. 프로젝트 실행 방법

```bash
# 1. 저장소 복제
git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)

# 2. 폴더로 이동
cd heart-disease-prediction

# 3. 필요한 라이브러리 설치
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 4. 주피터 노트북 실행
jupyter notebook Heart_Disease_Prediction.ipynb