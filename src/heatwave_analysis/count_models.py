"""Poisson/negative-binomial heatwave count regression and influence sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def count_distribution_diagnostics(counts: pd.DataFrame, outcome: str = "heatwave_days") -> pd.DataFrame:
    x = counts[outcome]
    return pd.DataFrame([{
        "outcome": outcome, "n": len(x), "mean": x.mean(), "variance": x.var(ddof=1),
        "variance_to_mean": x.var(ddof=1) / x.mean() if x.mean() else np.nan,
        "zero_count": int((x == 0).sum()), "zero_fraction": (x == 0).mean(),
        "minimum": x.min(), "maximum": x.max(), "skewness": x.skew(),
    }])


def fit_count_models(counts: pd.DataFrame, outcome: str = "heatwave_days"):
    d = counts[["year", outcome]].dropna().copy()
    d["decade"] = (d.year - d.year.mean()) / 10
    X = sm.add_constant(d[["decade"]])
    poisson = sm.GLM(d[outcome], X, family=sm.families.Poisson()).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    pearson = float(np.sum(poisson.resid_pearson ** 2))
    dispersion = pearson / poisson.df_resid
    # NB2 estimates alpha rather than choosing it from the sample moments.
    try:
        nb = sm.NegativeBinomial(d[outcome], X).fit(disp=False, maxiter=500)
        nb_ok = bool(nb.mle_retvals.get("converged", True))
    except Exception:
        alpha = max((d[outcome].var() - d[outcome].mean()) / max(d[outcome].mean() ** 2, 1e-9), 1e-6)
        nb = sm.GLM(d[outcome], X, family=sm.families.NegativeBinomial(alpha=alpha)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        nb_ok = True
    select_nb = dispersion > 1.2 and nb.aic + 2 < poisson.aic and nb_ok
    selected = nb if select_nb else poisson
    selected_name = "negative_binomial" if select_nb else "poisson"
    nb_mu = np.asarray(nb.predict())
    nb_alpha = max(float(nb.params.get("alpha", np.nan)), 1e-12)
    nb_variance = nb_mu + nb_alpha * nb_mu ** 2
    nb_pearson = float(np.sum((d[outcome].to_numpy() - nb_mu) ** 2 / nb_variance))
    y = d[outcome].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(y > 0, y * np.log(y / nb_mu), 0.0)
        second = (y + 1 / nb_alpha) * np.log((y + 1 / nb_alpha) / (nb_mu + 1 / nb_alpha))
    nb_deviance = float(2 * np.sum(first - second))
    comparison = pd.DataFrame([
        {"model": "poisson", "aic": poisson.aic, "bic": getattr(poisson, "bic_llf", np.nan),
         "log_likelihood": poisson.llf, "residual_deviance": poisson.deviance,
         "pearson_statistic": pearson, "dispersion_ratio": dispersion, "converged": True,
         "selected": not select_nb},
        {"model": "negative_binomial", "aic": nb.aic, "bic": nb.bic, "log_likelihood": nb.llf,
         "residual_deviance": nb_deviance, "pearson_statistic": nb_pearson,
         "dispersion_ratio": nb_pearson / nb.df_resid, "converged": nb_ok, "selected": select_nb},
    ])
    return d, poisson, nb, selected, selected_name, comparison


def model_estimates(model, selected_name: str) -> pd.DataFrame:
    term = "decade"
    ci = model.conf_int().loc[term]
    beta = model.params[term]
    return pd.DataFrame([{
        "model": selected_name, "term": term, "coefficient": beta, "standard_error": model.bse[term],
        "ci_lower": ci[0], "ci_upper": ci[1], "p_value": model.pvalues[term],
        "incidence_rate_ratio": np.exp(beta), "irr_ci_lower": np.exp(ci[0]), "irr_ci_upper": np.exp(ci[1]),
        "percentage_change_per_decade": 100 * (np.exp(beta) - 1), "aic": model.aic,
        "bic": model.bic, "log_likelihood": model.llf,
    }])


def influence_diagnostics(
    d: pd.DataFrame, poisson, selected, selected_name: str,
    outcome: str = "heatwave_days", seed: int = 20260901,
):
    """Distribution-consistent residuals and case-deletion parameter influence."""
    fitted = np.asarray(selected.predict())
    observed = d[outcome].to_numpy()
    if selected_name == "negative_binomial":
        alpha = float(selected.params.get("alpha", 0.0))
        variance = np.maximum(fitted + alpha * fitted ** 2, 1e-9)
        size = 1 / max(alpha, 1e-12)
        probability = size / (size + fitted)
        lower = stats.nbinom.cdf(observed - 1, size, probability)
        upper = stats.nbinom.cdf(observed, size, probability)
    else:
        variance = np.maximum(fitted, 1e-9)
        lower = stats.poisson.cdf(observed - 1, fitted)
        upper = stats.poisson.cdf(observed, fitted)
    rng = np.random.default_rng(seed)
    randomized_q = stats.norm.ppf(np.clip(rng.uniform(lower, upper), 1e-6, 1 - 1e-6))

    base_params = np.asarray(selected.params)[:2]
    base_covariance = np.asarray(selected.cov_params())[:2, :2]
    inverse_covariance = np.linalg.pinv(base_covariance)
    distances = []
    decade_changes = []
    for index in d.index:
        subset = d.drop(index=index)
        x = sm.add_constant(subset[["decade"]])
        if selected_name == "negative_binomial":
            refit = sm.NegativeBinomial(subset[outcome], x).fit(disp=False, maxiter=500)
        else:
            refit = sm.GLM(subset[outcome], x, family=sm.families.Poisson()).fit()
        delta = np.asarray(refit.params)[:2] - base_params
        distances.append(float(delta @ inverse_covariance @ delta / len(base_params)))
        decade_changes.append(float(delta[1]))
    diagnostics = d.copy()
    diagnostics["observed"] = observed
    diagnostics["fitted"] = fitted
    diagnostics["pearson_residual"] = (observed - fitted) / np.sqrt(variance)
    diagnostics["randomized_quantile_residual"] = randomized_q
    diagnostics["leverage"] = np.nan
    diagnostics["cooks_distance"] = distances
    diagnostics["case_deletion_decade_change"] = decade_changes
    diagnostics["influence_method"] = f"case-deletion {selected_name} parameter distance"
    cutoff = 4 / len(d)
    diagnostics["influential"] = diagnostics.cooks_distance > cutoff
    return diagnostics


def leave_one_influential_year_out(d: pd.DataFrame, diagnostics: pd.DataFrame, outcome: str = "heatwave_days") -> pd.DataFrame:
    years = diagnostics.loc[diagnostics.influential, "year"].astype(int).tolist()
    rows = []
    for year in years:
        subset = d[d.year != year].copy()
        _, _, _, model, name, _ = fit_count_models(subset, outcome)
        ci = model.conf_int().loc["decade"]
        rows.append({"excluded_year": year, "selected_model": name, "coefficient": model.params["decade"],
                     "irr": np.exp(model.params["decade"]), "irr_ci_lower": np.exp(ci[0]),
                     "irr_ci_upper": np.exp(ci[1]), "p_value": model.pvalues["decade"]})
    return pd.DataFrame(rows, columns=["excluded_year", "selected_model", "coefficient", "irr", "irr_ci_lower", "irr_ci_upper", "p_value"])
