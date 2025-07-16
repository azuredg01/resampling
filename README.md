# oversampling_v1.py 使用說明

## ✨ 主要功能
本腳本可針對不平衡資料集進行多種 over-sampling（過採樣）方法，支援 imbalanced-learn 套件的多種演算法。

## 🧩 支援方法
- 數值型：
  - RandomOverSampler
  - SMOTE
  - ADASYN
  - BorderlineSMOTE
  - SVMSMOTE
  - KMeansSMOTE
- 類別型：
  - SMOTEN
- 混合型（數值+類別）：
  - SMOTENC（需指定類別特徵）

## 🛠️ 基本用法
```bash
python oversampling_v1.py -m <方法名稱> -in <輸入CSV檔案> -out <輸出CSV檔案> [其他參數]
```

### 參數說明
- `-m, --method`：過採樣方法名稱（必填）
- `-in, --input`：輸入 .csv 檔案（必填，最後一欄為標籤，且資料必須為二維 (n_samples, n_features)）
- `-out, --output`：輸出 .csv 檔案（必填）
- `-ss, --sampling_strategy`：採樣策略 (要補至多少樣本數)（可選，預設 auto，可為 float、dict、str）
- `-cf, --categorical_features`：SMOTENC 專用，指定類別特徵欄位（如 -cf 1 4）

### 範例
#### 數值型資料（SMOTE）
```bash
python oversampling_v1.py -m SMOTE -in numeric_data.csv -out SMOTE.csv
```

#### 類別型資料（SMOTEN）
```bash
python oversampling_v1.py -m SMOTEN -in categorical_data.csv -out SMOTEN.csv
```

#### 混合型資料（SMOTENC，需指定類別特徵的欄位 `-cf`）
```bash
python oversampling_v1.py -m SMOTENC -in mixed_data.csv -out SMOTENC.csv -cf 1 4
```

#### 指定採樣數量（`-ss`，預設 auto）
```bash
python oversampling_v1.py -m RandomOverSampler -in numeric_data.csv -out ROS_dict.csv -ss "{0: 200, 1: 300}"
python oversampling_v1.py -m RandomOverSampler -in numeric_data.csv -out ROS_float.csv -ss 0.5
```
> • dict 可指定每個類別要補到的樣本數，例如 `-ss "{0: 200, 1: 300}"`。

## ⚠️ 注意事項
- 目前輸入檔案的格式為 .csv，且最後一欄為標籤。
- 輸入資料必須為二維 (n_samples, n_features)，不可為一維或僅一欄。
- SMOTENC 必須指定 `-cf` 參數。

---

如需更多細節，請參考程式內註解或 [imbalanced-learn 官方文件](https://imbalanced-learn.org/stable/references/over_sampling.html)。

