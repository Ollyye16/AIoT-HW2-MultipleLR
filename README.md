# HW2 — Multiple Linear Regression：預測各國平均壽命

物聯網應用與資料分析 HW2 · 依 CRISP-DM 流程完成多元線性迴歸、特徵選擇與模型評估。

- **Web（GitHub Pages）**：`https://<你的帳號>.github.io/<repo 名稱>/` ← 開啟 Pages 後填入
- **資料集**：[Life Expectancy (WHO) — Kaggle, kumarajarshi](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)（2,938 筆 × 22 欄，20 個可用特徵）
  - repo 內副本：[`data/Life_Expectancy_Data.csv`](data/Life_Expectancy_Data.csv)（SHA-256 `872125dd…87963b11`，與 Kaggle 原檔相同）

**網頁功能（加分項）**
- ✅ 下拉選單切換 problem：Life Expectancy (WHO)、50 Startups、Boston Housing
- ✅ 可貼上任意 CSV 網址當資料集，自動辨識數值/類別欄位、處理缺值
- 5 種特徵選擇方法 × k = 1…p 比較、係數檢定、**95% 信賴區間 / 預測區間圖**、即時預測、殘差診斷

---

## 1. Business Understanding（商業理解）

世界衛生組織（WHO）與各國政府需要知道「哪些可控因素最能影響國民平均壽命」，才能決定預算該優先投入在疫苗、教育、醫療支出或疾病防治。

- **目標**：用各國每年的健康、經濟、社會指標，預測該年的平均壽命（`Life expectancy`，單位：歲）。
- **問題類型**：連續數值的迴歸預測 → 多元線性迴歸。
- **成功標準**：測試集 R² > 0.8、RMSE 盡量小，而且模型係數要能解釋（哪些因素正向/負向影響壽命）。

## 2. Data Understanding（資料理解）

| 項目 | 內容 |
|---|---|
| 筆數 | 2,938（193 個國家 × 2000–2015 年） |
| 目標 y | `Life expectancy`（10 筆缺值） |
| 特徵 | 20 個：`Year`、`Status`（開發中/已開發）、`Adult Mortality`、`infant deaths`、`Alcohol`、`percentage expenditure`、`Hepatitis B`、`Measles`、`BMI`、`under-five deaths`、`Polio`、`Total expenditure`、`Diphtheria`、`HIV/AIDS`、`GDP`、`Population`、`thinness 1-19 years`、`thinness 5-9 years`、`Income composition of resources`、`Schooling` |
| 不使用 | `Country`（193 個類別，屬於識別欄位） |

**缺值**最多的欄位：Population 652、Hepatitis B 553、GDP 448、Total expenditure 226、Alcohol 194。

**與壽命的相關係數**（[`figures/correlation.png`](figures/correlation.png)）：

| 最強正相關 | r | 最強負相關 | r |
|---|---|---|---|
| Schooling | 0.752 | Adult Mortality | −0.696 |
| Income composition of resources | 0.725 | HIV/AIDS | −0.557 |
| BMI | 0.568 | thinness 1-19 years | −0.477 |

另外，`infant deaths` 與 `under-five deaths` 幾乎完全共線（VIF ≈ 200），`thinness 1-19` 與 `thinness 5-9` 也高度相關。

![correlation](figures/correlation.png)

## 3. Data Preparation（資料準備）

1. **欄名清理**：原始欄名有前後空白（如 `"Life expectancy "`），全部 trim。
2. **刪除目標缺值**：`Life expectancy` 缺值的 10 筆刪除 → 2,928 筆。
3. **切分**：固定亂數種子（seed = 42）隨機切 80% 訓練（2,342 筆）/ 20% 測試（586 筆）。
4. **缺值補值**：用**訓練集的中位數**補（共 2,513 格），測試集也用訓練集的中位數，避免資料洩漏。
5. **類別編碼**：`Status` 做 one-hot，以 `Developed` 為基準，產生 `Status_Developing`。
6. **標準化**：只用在 Lasso / ElasticNet / RFE 的特徵排序；最終 OLS 模型使用原始單位，係數才好解釋。

## 4. Modeling（建模）

模型：**OLS 多元線性迴歸**（statsmodels）。特徵選擇只用訓練集，比較 5 種方法在 k = 1…20 個特徵下的表現：

| 方法 | 做法 |
|---|---|
| Lasso | L1 正則化路徑，依特徵「進場」先後排序（`sklearn.linear_model.lasso_path`） |
| ElasticNet | L1 + L2（l1_ratio = 0.5）正則化路徑排序（`enet_path`） |
| Forward SFS | 每次加入讓訓練集 SSE 下降最多的特徵 |
| RFE | 從全部特徵開始，反覆移除標準化係數最小的特徵（`sklearn.feature_selection.RFE`） |
| Exhaustive | 窮舉所有子集；p = 20 時有 2²⁰ 種組合，改用 Forward SFS 的結果 |

![rmse by k](figures/rmse_r2_by_k.png)

## 5. Evaluation（評估）

**各方法的最佳 k（測試集）**

| 方法 | 最佳 k | RMSE（歲） | R² | Adj. R² | MAE |
|---|---|---|---|---|---|
| **RFE** | **16** | **4.064** | **0.8288** | **0.8240** | **3.046** |
| Forward SFS | 18 | 4.073 | 0.8281 | 0.8227 | 3.051 |
| Lasso | 19 | 4.073 | 0.8281 | 0.8223 | 3.052 |
| ElasticNet | 19 | 4.073 | 0.8281 | 0.8223 | 3.052 |

**重點觀察**
- **邊際效益遞減**：Forward SFS 只用 4 個特徵（Schooling、Adult Mortality、HIV/AIDS、Diphtheria）就有 RMSE 4.511、R² 0.789。再加 12 個特徵，RMSE 只再降 0.45 歲。如果重視可解釋性，4 特徵模型是很好的精簡版本。
- **共線性會誤導 RFE**：RFE 依係數大小排序。`infant deaths` 和 `under-five deaths` 高度共線，係數被放大成一正一負，所以 RFE 最先保留它們，導致 k = 1、2 時 RMSE 高達 9.6、9.0。但兩者一起放進完整模型時確實有用（k = 16 最佳）。
- **最佳模型 RFE k = 16 的係數**（[`figures/best_model_coefficients.csv`](figures/best_model_coefficients.csv)）：教育年數 Schooling（+0.65 歲/年）、收入組成指數（+4.76）、BMI、疫苗接種率（Diphtheria、Polio）為正向；成人死亡率、HIV/AIDS（−0.44）、開發中國家（−1.75 歲）為負向。Alcohol 與 Total expenditure 不顯著（p ≈ 0.16）。

**預測圖（95% 信賴區間 + 預測區間）**

![prediction interval](figures/prediction_interval.png)

- 深色帶 = **信賴區間（CI）**：「平均」壽命的不確定性，很窄，代表係數估得很準。
- 淺色帶 = **預測區間（PI）**：「單一國家某年」壽命的範圍，約 ±8 歲。
- 測試集有 **549 / 586 = 93.7%** 的實際值落在 95% PI 內，接近理論值 95%。區間外的點大多是低壽命國家，而且落在預測值下方，表示模型對這些國家有高估的情形。

### 5.1 迴歸五大假設檢定（最佳模型，訓練集殘差，α = 0.05）

| # | 假設 | 檢定 | 結果 | 判定 |
|---|---|---|---|---|
| ① | 線性 | Ramsey RESET（加入 ŷ²） | p < 0.0001 | ⚠ 有未捕捉的非線性 |
| ② | 誤差獨立 | Durbin-Watson | DW = 0.76 | ⚠ 正自相關 |
| ③ | 同變異 | Breusch-Pagan | LM = 298.2，p < 0.0001 | ⚠ 異質變異 |
| ④ | 殘差常態 | Jarque-Bera / Shapiro-Wilk | JB = 280.0，p < 0.0001；偏態 −0.21、峰態 4.64 | ⚠ 尾巴偏厚 |
| ⑤ | 無嚴重共線性 | VIF | 最大 200.8（infant deaths ↔ under-five deaths） | ⚠ 共線性嚴重 |

![assumptions](figures/assumptions.png)

**解讀**
- **②獨立**：資料依「國家 → 年份」排序，同一個國家連續 16 年的誤差彼此相關（某國被高估，往往每年都被高估），所以 DW 遠低於 2。這是 panel data 的特性；更嚴謹的做法是用國家固定效果或 cluster-robust 標準誤。
- **③同變異**：把殘差依預測值分組，預測壽命 < 65 歲的國家殘差標準差 4.6–5.3 歲，> 72 歲只有 3.1–3.3 歲。低壽命國家的不確定性較大，所以這些國家的預測區間其實應該更寬。
- **④常態**：Q-Q 圖中段貼近直線，偏離集中在左尾（少數嚴重高估的國家）。樣本數 2,342 很大，檢定對小偏離也很敏感。
- **影響**：違反假設時，OLS 係數的點估計仍然不偏，所以「哪些因素正向/負向」的結論可信；但 p 值、信賴區間與預測區間會偏窄，要保守解讀。網頁上每選一個模型都會即時重新檢定（JS 與 statsmodels 數值一致）。

### 5.2 交叉驗證：最佳 k 穩不穩？

單一測試集的 RMSE 可能只是剛好切到好分的資料，所以在訓練集上再做 5-fold 交叉驗證：

![cv by k](figures/cv_by_k.png)

- CV 曲線和測試集曲線形狀一致：k = 1→5 快速下降，之後趨於平緩。RFE 從 k = 12 起 CV RMSE 都在 **4.07–4.09** 之間；Forward SFS 與 Lasso 在 k = 13–16 約 4.15，要到 k ≥ 17–18 才降到 4.08。
- RFE 的 CV 最低點在 k = 15（4.074），測試集最低點在 k = 16（4.064），兩者只差 0.01 歲。結論是「13 個以上特徵效果都差不多」，k = 16 不是靠運氣選到的。

### 5.3 OLS vs Ridge vs Lasso

| 模型 | 測試 RMSE | MAE | MSE | R² | 5-fold CV RMSE | α（CV 選擇） |
|---|---|---|---|---|---|---|
| **OLS（RFE k = 16）** | **4.064** | **3.046** | **16.52** | **0.8288** | 4.081 | — |
| OLS（全部 20 個） | 4.074 | 3.053 | 16.60 | 0.8280 | 4.086 | — |
| Ridge（全部 20 個） | 4.077 | 3.054 | 16.62 | 0.8277 | 4.085 | 0.349 |
| Lasso（全部 20 個） | 4.091 | 3.061 | 16.73 | 0.8266 | 4.087 | 0.0067 |

正則化**沒有帶來改善**：交叉驗證選出的 α 都很小，Lasso 也保留了全部 20 個係數。原因是樣本數（2,342）遠大於特徵數（20），OLS 本來就沒有明顯過度擬合，正則化能壓的變異很少。

### 5.4 模型改善：log 轉換 + 影響點

| 版本 | 訓練筆數 | 測試 RMSE | 改善 | R² |
|---|---|---|---|---|
| ① 原始特徵（全部 20 個） | 2,342 | 4.074 | — | 0.8280 |
| **② 對偏態特徵取 log1p** | 2,342 | **3.619** | **−11.2%** | **0.8643** |
| ③ log1p + 移除 Cook's D > 4/n | 2,200 | 3.636 | −10.7% | 0.8630 |

- 有 10 個特徵偏態 > 2（GDP、Population、HIV/AIDS、percentage expenditure、Measles、infant/under-five deaths、Hepatitis B、Polio、Diphtheria）。例如 Population 中位數 147 萬、最大值 12.9 億，偏態 17.9。取 log 後關係更接近線性，**RMSE 降了 11%，是所有嘗試中最有效的改善**，比挑特徵或正則化都有用。
- 用 Cook's distance 找出 142 個影響點，移除後測試 RMSE 反而略差，表示這些點帶有測試集也需要的資訊。沒有證據顯示它們是資料錯誤，所以不刪除。

![cooks distance](figures/cooks_distance.png)

### 5.5 實作時踩到的坑

一開始用 scikit-learn 的 `LinearRegression` 直接擬合全部 20 個原始特徵，測試 RMSE 是 6.80；同一個模型用 statsmodels 卻是 4.07。原因是 `Population`（~10⁹）和比例型特徵（0–1）尺度差太多，最小平方法把太小的奇異值當成 0 截掉，模型就悄悄變差。**先標準化再擬合**（數學上是同一個模型）就恢復成 4.07。這也提醒我們：不同工具的結果要交叉核對。

### 5.6 結論

1. **關鍵因素**：教育年數（Schooling）、收入組成指數是延長壽命最強的正向因素；成人死亡率、HIV/AIDS 是最強的負向因素；其他條件相同時，開發中國家約少 1.75 歲。
2. **模型表現**：OLS 測試 R² 0.83、RMSE 4.06 歲；對偏態特徵取 log 後 R² 提升到 0.86、RMSE 3.62 歲。
3. **精簡模型**：只用 4 個特徵就有 R² 0.79，適合需要簡單解釋的情境。
4. **限制**：資料是 panel 結構（同國家多年）且有異質變異，p 值與區間偏樂觀；未來可用固定效果模型或 cluster-robust 標準誤改善。

## 6. Deployment（部署）

| 檔案 | 說明 |
|---|---|
| `index.html` | 網頁前端（Chart.js）。後端連得到就用 Python 計算；連不到（GitHub Pages）時改用瀏覽器內 JavaScript，結果相同 |
| `app.py` | Flask 後端：`POST /api/analyze`，scikit-learn + statsmodels |
| `analysis.py` | 離線產生報告用的圖表與數據（`figures/`），包含五大假設、交叉驗證、Ridge/Lasso、log 轉換與 Cook's distance；`figures/advanced.json` 供網頁「評估」區使用 |
| `infographic.html` | 一頁式資訊圖表（[開啟](infographic.html)） |
| `api/index.py`、`vercel.json` | Vercel 部署設定（選用） |

```bash
pip install -r requirements.txt matplotlib
python app.py          # 開 http://localhost:5000/
python analysis.py     # 重新產生 figures/（需要 matplotlib）
```

---

## GPT 輔助內容

> TODO（同學自行填寫）：說明哪些部分有使用 ChatGPT / Claude 協助（例如程式撰寫、統計解釋），並附上對話 PDF 連結（pdfCrowd 匯出）。

## NotebookLM 摘要

> TODO（同學自行填寫）：用 NotebookLM 研究網路上同主題（WHO Life Expectancy 線性迴歸）的解法，撰寫 100 字以上摘要。
