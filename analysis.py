"""HW2 offline analysis for the report: runs the same pipeline as the web app (app.analyze)
on the Life Expectancy dataset, then the extra evaluation the web page cannot do in the browser
(regression assumption tests, Ridge/Lasso with 5-fold CV, log transforms, Cook's distance).
Writes figures + tables to ./figures and figures/advanced.json for the web page.

    python analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from app import analyze, prepare

DATA = "data/Life_Expectancy_Data.csv"
TARGET = 3  # "Life expectancy"
OUT = Path("figures")
NAMES = {"lasso": "Lasso", "forward": "Forward SFS", "rfe": "RFE", "enet": "ElasticNet", "exhaustive": "Exhaustive*"}
CV = KFold(5, shuffle=True, random_state=42)
# Always standardize before sklearn OLS: with raw Population (~1e9) next to 0-1 ratios, lstsq drops small
# singular values and the fit silently degrades (test RMSE 6.8 instead of 4.07 for the same model).
OLS = lambda: make_pipeline(StandardScaler(), LinearRegression())


def metrics(y, yh):
    e = y - yh
    return {"mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def cv_rmse(model, X, y):
    return float(-cross_val_score(model, X, y, cv=CV, scoring="neg_root_mean_squared_error").mean())


def assumptions(X, y):
    """Five classic OLS assumptions on the TRAINING fit."""
    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, Xc).fit()
    e = ols.resid
    jb, jbp, skew, kurt = jarque_bera(e)
    bp, bpp, _, _ = het_breuschpagan(e, Xc)
    sh, shp = stats.shapiro(e[:5000])
    # linearity: Ramsey RESET-style check — does adding ŷ² improve the fit?
    reset = sm.OLS(y, np.column_stack([Xc, ols.fittedvalues ** 2])).fit()
    lin_p = float(reset.pvalues[-1])
    return ols, {
        "linearity_reset_p": lin_p,
        "normality_jb": float(jb), "normality_jb_p": float(jbp), "skew": float(skew), "kurtosis": float(kurt),
        "normality_shapiro_p": float(shp),
        "homoscedasticity_bp": float(bp), "homoscedasticity_bp_p": float(bpp),
        "independence_dw": float(durbin_watson(e)),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    OUT.mkdir(exist_ok=True)

    # ================= Data Understanding =================
    df = pd.read_csv(DATA, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    print(f"shape={df.shape}  countries={df['Country'].nunique()}  years={df['Year'].min()}-{df['Year'].max()}")
    miss = df.isna().sum()
    print("missing values:\n" + miss[miss > 0].sort_values(ascending=False).to_string())
    corr = df.select_dtypes("number").corr()["Life expectancy"].drop("Life expectancy").sort_values()
    print("correlation with Life expectancy:\n" + corr.round(3).to_string())
    corr.plot.barh(figsize=(7, 6), color=np.where(corr > 0, "#0f766e", "#be123c"), title="Correlation with Life expectancy")
    plt.tight_layout(); plt.savefig(OUT / "correlation.png", dpi=150); plt.close()

    # ================= Modeling: feature selection (same code path as the Flask API) =================
    r = analyze(DATA, TARGET)
    feats, res = r["features"], r["results"]
    rows = []
    for m, L in res.items():
        for x in L:
            rows.append({"method": NAMES[m], "k": x["k"], "rmse": x["rmse"], "r2": x["r2"], "adj_r2": x["adj"], "mae": x["mae"],
                         "features": ", ".join(feats[j] for j in x["F"])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results_all.csv", index=False)
    best = table.loc[table.groupby("method")["rmse"].idxmin()].sort_values("rmse")
    print("\nbest k per method:\n" + best[["method", "k", "rmse", "r2", "adj_r2", "mae"]].round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, L in res.items():
        ax[0].plot([x["k"] for x in L], [x["rmse"] for x in L], marker="o", ms=3, label=NAMES[m])
        ax[1].plot([x["k"] for x in L], [x["r2"] for x in L], marker="o", ms=3, label=NAMES[m])
    ax[0].set(title="Test RMSE by number of features", xlabel="k", ylabel="RMSE (years)")
    ax[1].set(title="Test R² by number of features", xlabel="k", ylabel="R²")
    for a in ax:
        a.set_xticks(range(1, len(feats) + 1))
    ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "rmse_r2_by_k.png", dpi=150); plt.close()

    # ---- best model + prediction plot with 95% CI / PI ----
    m_best, x_best = min(((m, x) for m, L in res.items() for x in L), key=lambda t: t[1]["rmse"])
    _, Xtr, ytr, Xte, yte, _ = prepare(DATA, TARGET)
    F = x_best["F"]
    o = np.argsort(x_best["yhTe"])
    s = lambda key: np.asarray(x_best[key])[o]
    xs = np.arange(1, len(o) + 1)
    inside = ((yte >= np.asarray(x_best["piLo"])) & (yte <= np.asarray(x_best["piHi"]))).mean()
    plt.figure(figsize=(11, 5))
    plt.fill_between(xs, s("piLo"), s("piHi"), color="#2dd4bf", alpha=.18, label="95% prediction interval")
    plt.fill_between(xs, s("ciLo"), s("ciHi"), color="#0f766e", alpha=.6, label="95% confidence interval")
    plt.plot(xs, s("yhTe"), color="#0f766e", lw=1.5, label="prediction ŷ")
    plt.scatter(xs, yte[o], s=6, color="#b45309", label="actual y", zorder=3)
    plt.title(f"{NAMES[m_best]} k={x_best['k']}: test RMSE={x_best['rmse']:.3f}, R²={x_best['r2']:.4f}, PI coverage={inside:.1%}")
    plt.xlabel("test samples (sorted by prediction)"); plt.ylabel("Life expectancy (years)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "prediction_interval.png", dpi=150); plt.close()

    coef = pd.DataFrame({"feature": ["(Intercept)"] + [feats[j] for j in F], "beta": x_best["b"], "se": x_best["se"],
                         "t": x_best["t"], "p": x_best["p"], "ci_low": x_best["lo"], "ci_high": x_best["hi"],
                         "vif": [np.nan] + x_best["vif"]})
    coef.to_csv(OUT / "best_model_coefficients.csv", index=False)
    print(f"\nbest model: {NAMES[m_best]} k={x_best['k']}  PI coverage={inside:.3f}\n" + coef.round(4).to_string(index=False))

    # ================= Evaluation 1: five regression assumptions =================
    ols, A = assumptions(Xtr[:, F], ytr)
    A["max_vif"] = float(max(x_best["vif"]))
    print("\nassumptions:", json.dumps({k: round(v, 4) for k, v in A.items()}))
    e, fit = ols.resid, ols.fittedvalues
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].scatter(fit, e, s=4, alpha=.5, color="#0f766e"); ax[0, 0].axhline(0, color="#b45309")
    ax[0, 0].set(title="① Linearity: residuals vs fitted", xlabel="fitted", ylabel="residual")
    sm.qqplot(e / e.std(), line="45", ax=ax[0, 1], markersize=2, alpha=.5)
    ax[0, 1].set_title(f"④ Normality: Q-Q plot (JB p={A['normality_jb_p']:.2g})")
    ax[1, 0].scatter(fit, np.sqrt(np.abs(e / e.std())), s=4, alpha=.5, color="#7c3aed")
    ax[1, 0].set(title=f"③ Homoscedasticity: scale-location (BP p={A['homoscedasticity_bp_p']:.2g})", xlabel="fitted", ylabel="√|std residual|")
    ax[1, 1].hist(e, bins=50, color="#0f766e", alpha=.8)
    ax[1, 1].set(title=f"② Independence: residual histogram (Durbin-Watson={A['independence_dw']:.2f})", xlabel="residual")
    plt.tight_layout(); plt.savefig(OUT / "assumptions.png", dpi=150); plt.close()

    # ================= Evaluation 2: 5-fold CV — is the chosen k robust? =================
    cv_k = {}
    for m in ("forward", "rfe", "lasso"):
        cv_k[m] = [cv_rmse(OLS(), Xtr[:, L["F"]], ytr) for L in res[m]]
    plt.figure(figsize=(8, 4.5))
    for m, v in cv_k.items():
        plt.plot(range(1, len(v) + 1), v, marker="o", ms=3, label=f"{NAMES[m]} (5-fold CV)")
    plt.plot([x["k"] for x in res[m_best]], [x["rmse"] for x in res[m_best]], "k--", lw=1, label=f"{NAMES[m_best]} (test set)")
    plt.xticks(range(1, len(feats) + 1)); plt.xlabel("k"); plt.ylabel("RMSE (years)")
    plt.title("5-fold cross-validated RMSE on the training set"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "cv_by_k.png", dpi=150); plt.close()
    print("\nCV RMSE by k:", {m: [round(v, 3) for v in L] for m, L in cv_k.items()})
    # where does CV jump? show the feature added at that step and the per-fold RMSE
    for m, v in cv_k.items():
        for i in [i for i in range(1, len(v)) if v[i] > v[i - 1] * 1.2]:
            added = set(res[m][i]["F"]) - set(res[m][i - 1]["F"])
            folds = -cross_val_score(OLS(), Xtr[:, res[m][i]["F"]], ytr, cv=CV, scoring="neg_root_mean_squared_error")
            print(f"  {NAMES[m]} k={i + 1}: CV RMSE {v[i - 1]:.3f} -> {v[i]:.3f} after adding {[feats[j] for j in added]}; folds={np.round(folds, 2).tolist()}")
    for name in ("Population", "percentage expenditure"):
        j = feats.index(name)
        print(f"  {name}: median={np.median(Xtr[:, j]):.0f}  max={Xtr[:, j].max():.0f}  skew={stats.skew(Xtr[:, j]):.1f}")

    # ================= Evaluation 3: OLS vs Ridge vs Lasso =================
    alphas = np.logspace(-3, 3, 60)
    models = {
        f"OLS ({NAMES[m_best]} k={len(F)})": (OLS(), F),
        "OLS (all 20)": (OLS(), list(range(len(feats)))),
        "Ridge (all 20, CV α)": (make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=CV)), list(range(len(feats)))),
        "Lasso (all 20, CV α)": (make_pipeline(StandardScaler(), LassoCV(cv=CV, random_state=0, max_iter=50000)), list(range(len(feats)))),
    }
    compare = []
    for name, (mdl, cols) in models.items():
        mdl.fit(Xtr[:, cols], ytr)
        row = {"model": name, **metrics(yte, mdl.predict(Xte[:, cols])), "cv_rmse": cv_rmse(mdl, Xtr[:, cols], ytr)}
        last = mdl[-1]
        if hasattr(last, "alpha_"):
            row["alpha"] = float(last.alpha_)
            row["nonzero"] = int((np.abs(last.coef_) > 1e-8).sum())
        compare.append(row)
    compare = pd.DataFrame(compare)
    compare.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nmodel comparison:\n" + compare.round(4).to_string(index=False))

    # ================= Evaluation 4: log transforms + influential points =================
    skewed = [j for j in range(len(feats)) if abs(stats.skew(Xtr[:, j])) > 2 and Xtr[:, j].min() >= 0]
    print("\nlog1p on skewed features:", [feats[j] for j in skewed])
    logX = lambda X: np.column_stack([np.log1p(X[:, j]) if j in skewed else X[:, j] for j in range(X.shape[1])])
    Ltr, Lte = logX(Xtr), logX(Xte)
    allc = list(range(len(feats)))
    infl = sm.OLS(ytr, sm.add_constant(Ltr)).fit().get_influence()
    cooks = infl.cooks_distance[0]
    keep = cooks <= 4 / len(ytr)
    variants = [
        ("① 原始特徵（all 20）", Xtr, Xte, np.ones(len(ytr), bool)),
        ("② log1p 偏態特徵", Ltr, Lte, np.ones(len(ytr), bool)),
        (f"③ log1p + 移除 Cook's D > 4/n（{(~keep).sum()} 筆）", Ltr, Lte, keep),
    ]
    improve = []
    for name, A_tr, A_te, mask in variants:
        mdl = OLS().fit(A_tr[mask][:, allc], ytr[mask])
        improve.append({"variant": name, **metrics(yte, mdl.predict(A_te[:, allc])), "train_rows": int(mask.sum())})
    improve = pd.DataFrame(improve)
    improve.to_csv(OUT / "improvement.csv", index=False)
    print("\nimprovement:\n" + improve.round(4).to_string(index=False))
    plt.figure(figsize=(8, 4))
    plt.stem(cooks, markerfmt=" ", basefmt=" ")
    plt.axhline(4 / len(ytr), color="#be123c", ls="--", label="4/n")
    plt.title(f"Cook's distance (training set, log1p features): {(~keep).sum()} influential points")
    plt.xlabel("training sample"); plt.ylabel("Cook's D"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "cooks_distance.png", dpi=150); plt.close()

    # ================= export for web page / report =================
    out = {
        "best": {"method": NAMES[m_best], "k": x_best["k"], "features": [feats[j] for j in F], "rmse": x_best["rmse"],
                 "r2": x_best["r2"], "adj_r2": x_best["adj"], "mae": x_best["mae"], "pi_coverage": float(inside),
                 "n_train": int(len(ytr)), "n_test": int(len(yte))},
        "assumptions": A,
        "cv_by_k": cv_k,
        "model_comparison": json.loads(compare.to_json(orient="records")),  # NaN -> null (NaN is not valid JSON)
        "skewed_log1p": [feats[j] for j in skewed],
        "improvement": json.loads(improve.to_json(orient="records")),
    }
    json.dump(out, open(OUT / "advanced.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(out["best"], open(OUT / "best_model.json", "w"), indent=2)


if __name__ == "__main__":
    main()
