"""纯 Python 实现的公积金贷款抵押估值风险模型。"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DATASET_FILENAME = "sample_gjj_mortgage.csv"
TARGET_COLUMN = "collateral_value"
FEATURE_COLUMNS = [
    "loan_amount",
    "appraised_value",
    "monthly_income",
    "housing_fund_contribution",
    "employment_length_years",
    "credit_score",
    "delinquency_history",
    "co_borrower",
    "has_mortgage_insurance",
    "loan_term_months",
    "interest_rate",
]
DEFAULT_RISK_THRESHOLDS = {"low": 0.60, "medium": 0.80}


def poisson_sample(rng: random.Random, lam: float) -> int:
    """Sample from a Poisson distribution using the Knuth algorithm."""

    l_value = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_value:
        k += 1
        p *= rng.random()
    return max(k - 1, 0)


@dataclass
class LoanRecord:
    loan_id: int
    issue_date: str
    region: str
    property_type: str
    loan_amount: float
    appraised_value: float
    interest_rate: float
    loan_term_months: int
    borrower_age: int
    monthly_income: float
    housing_fund_contribution: float
    employment_length_years: int
    credit_score: int
    delinquency_history: int
    co_borrower: int
    has_mortgage_insurance: int
    collateral_value: float

    def to_row(self) -> Dict[str, str]:
        payload = asdict(self)
        for key, value in payload.items():
            payload[key] = str(value)
        return payload


@dataclass
class RiskModel:
    intercept: float
    coefficients: Dict[str, float]
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    feature_order: List[str]

    def predict(self, record: LoanRecord) -> float:
        score = self.intercept
        for feature in self.feature_order:
            mean = self.feature_means[feature]
            std = self.feature_stds[feature]
            if std == 0:
                std = 1.0
            value = getattr(record, feature)
            score += self.coefficients[feature] * ((value - mean) / std)
        return score


@dataclass
class TrainingArtifacts:
    model: RiskModel
    metrics: Dict[str, float]
    risk_report: List[Dict[str, float]]
    risk_summary: List[Dict[str, float]]


def create_synthetic_dataset(
    n_samples: int = 800,
    *,
    seed: int = 42,
) -> List[LoanRecord]:
    rng = random.Random(seed)
    regions = ["江北", "江南", "萧山", "滨江", "临平"]
    property_types = ["公寓", "排屋", "别墅", "商住两用"]

    records: List[LoanRecord] = []
    for idx in range(1, n_samples + 1):
        issue_date = f"201{rng.randint(7, 9)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        loan_amount = rng.uniform(320_000, 1_300_000)
        appraised_value = loan_amount * rng.uniform(1.0, 1.45)
        interest_rate = rng.uniform(2.6, 4.35)
        loan_term = rng.randint(120, 360)
        borrower_age = rng.randint(23, 55)
        monthly_income = rng.uniform(6_000, 36_000)
        housing_fund = monthly_income * rng.uniform(0.08, 0.12)
        employment_length = rng.randint(1, 30)
        credit_score = rng.randint(580, 860)
        delinquency_history = min(poisson_sample(rng, 0.35), 6)
        co_borrower = rng.randint(0, 1)
        has_insurance = rng.randint(0, 1)

        quality_factor = (
            0.55 * (appraised_value / 1_000_000)
            + 0.20 * (credit_score / 850)
            + 0.15 * (monthly_income / 20_000)
            + 0.10 * (1 - delinquency_history / 6)
        )
        location_factor = regions.index(rng.choice(regions)) * 0.015
        property_factor = property_types.index(rng.choice(property_types)) * 0.02
        noise = rng.gauss(0, 45_000)
        collateral_value = appraised_value * (0.92 + 0.08 * quality_factor)
        collateral_value *= (1 + location_factor)
        collateral_value *= (1 + property_factor)
        collateral_value += noise
        collateral_value = max(collateral_value, loan_amount * 0.9)

        records.append(
            LoanRecord(
                loan_id=idx,
                issue_date=issue_date,
                region=rng.choice(regions),
                property_type=rng.choice(property_types),
                loan_amount=round(loan_amount, 2),
                appraised_value=round(appraised_value, 2),
                interest_rate=round(interest_rate, 4),
                loan_term_months=loan_term,
                borrower_age=borrower_age,
                monthly_income=round(monthly_income, 2),
                housing_fund_contribution=round(housing_fund, 2),
                employment_length_years=employment_length,
                credit_score=credit_score,
                delinquency_history=delinquency_history,
                co_borrower=co_borrower,
                has_mortgage_insurance=has_insurance,
                collateral_value=round(collateral_value, 2),
            )
        )
    return records


def save_dataset(records: Sequence[LoanRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(LoanRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def load_dataset(path: Path, create_if_missing: bool = True) -> List[LoanRecord]:
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(path)
        records = create_synthetic_dataset()
        save_dataset(records, path)
        return records

    records: List[LoanRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            records.append(
                LoanRecord(
                    loan_id=int(row["loan_id"]),
                    issue_date=row["issue_date"],
                    region=row["region"],
                    property_type=row["property_type"],
                    loan_amount=float(row["loan_amount"]),
                    appraised_value=float(row["appraised_value"]),
                    interest_rate=float(row["interest_rate"]),
                    loan_term_months=int(row["loan_term_months"]),
                    borrower_age=int(row["borrower_age"]),
                    monthly_income=float(row["monthly_income"]),
                    housing_fund_contribution=float(row["housing_fund_contribution"]),
                    employment_length_years=int(row["employment_length_years"]),
                    credit_score=int(row["credit_score"]),
                    delinquency_history=int(row["delinquency_history"]),
                    co_borrower=int(row["co_borrower"]),
                    has_mortgage_insurance=int(row["has_mortgage_insurance"]),
                    collateral_value=float(row["collateral_value"]),
                )
            )
    return records


def compute_feature_stats(records: Sequence[LoanRecord]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for feature in FEATURE_COLUMNS:
        values = [getattr(record, feature) for record in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            std = 1.0
        stats[feature] = {"mean": mean, "std": std}
    return stats


def build_design_matrix(
    records: Sequence[LoanRecord],
    stats: Dict[str, Dict[str, float]],
) -> List[List[float]]:
    design: List[List[float]] = []
    for record in records:
        row: List[float] = []
        for feature in FEATURE_COLUMNS:
            feature_stats = stats[feature]
            row.append((getattr(record, feature) - feature_stats["mean"]) / feature_stats["std"])
        design.append(row)
    return design


def build_target_vector(records: Sequence[LoanRecord]) -> List[float]:
    return [record.collateral_value for record in records]


def solve_normal_equation(
    design: Sequence[Sequence[float]],
    target: Sequence[float],
) -> List[float]:
    rows = len(design)
    cols = len(design[0]) + 1

    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]

    for i in range(rows):
        features = [1.0] + list(design[i])
        y = target[i]
        for a in range(cols):
            xty[a] += features[a] * y
            for b in range(cols):
                xtx[a][b] += features[a] * features[b]

    return gaussian_elimination(xtx, xty)


def gaussian_elimination(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    augmented = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be solved")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]

    solution = [augmented[i][n] for i in range(n)]
    return solution


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def root_mean_square_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((value - mean_true) ** 2 for value in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def evaluate_model(model: RiskModel, records: Sequence[LoanRecord]) -> Dict[str, float]:
    y_true = [record.collateral_value for record in records]
    y_pred = [model.predict(record) for record in records]
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "rmse": round(root_mean_square_error(y_true, y_pred), 2),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def generate_risk_report(
    model: RiskModel,
    records: Sequence[LoanRecord],
    thresholds: Dict[str, float] | None = None,
) -> List[Dict[str, float]]:
    thresholds = thresholds or DEFAULT_RISK_THRESHOLDS
    low = thresholds.get("low", 0.6)
    medium = thresholds.get("medium", 0.8)

    report: List[Dict[str, float]] = []
    for record in records:
        predicted_value = model.predict(record)
        predicted_ltv = record.loan_amount / predicted_value if predicted_value else 1.0
        if predicted_ltv <= low:
            risk = "低风险"
        elif predicted_ltv <= medium:
            risk = "中风险"
        else:
            risk = "高风险"

        row = record.to_row()
        row.update(
            {
                "predicted_collateral_value": round(predicted_value, 2),
                "predicted_ltv": round(predicted_ltv, 4),
                "risk_level": risk,
            }
        )
        report.append(row)
    return report


def summarise_risk_levels(report: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {"低风险": [], "中风险": [], "高风险": []}
    for row in report:
        grouped.setdefault(row["risk_level"], []).append(row)

    summary: List[Dict[str, float]] = []
    for risk_level, rows in grouped.items():
        if not rows:
            continue
        sample_size = len(rows)
        avg_ltv = sum(float(r["predicted_ltv"]) for r in rows) / sample_size
        avg_loan = sum(float(r["loan_amount"]) for r in rows) / sample_size
        overdue_rate = sum(1 for r in rows if int(r["delinquency_history"]) > 0) / sample_size
        summary.append(
            {
                "risk_level": risk_level,
                "sample_size": sample_size,
                "avg_predicted_ltv": round(avg_ltv, 4),
                "avg_loan_amount": round(avg_loan, 2),
                "overdue_rate": round(overdue_rate, 4),
            }
        )
    summary.sort(key=lambda item: item["avg_predicted_ltv"])
    return summary


def train_risk_model(dataset_path: Path) -> TrainingArtifacts:
    records = load_dataset(dataset_path)
    shuffled = list(records)
    random.Random(2023).shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_records = shuffled[:split_idx]
    test_records = shuffled[split_idx:]

    stats = compute_feature_stats(train_records)
    design_matrix = build_design_matrix(train_records, stats)
    target_vector = build_target_vector(train_records)

    weights = solve_normal_equation(design_matrix, target_vector)
    intercept, coef_values = weights[0], weights[1:]
    coefficients = {feature: coef_values[idx] for idx, feature in enumerate(FEATURE_COLUMNS)}

    model = RiskModel(
        intercept=intercept,
        coefficients=coefficients,
        feature_means={feature: stats[feature]["mean"] for feature in FEATURE_COLUMNS},
        feature_stds={feature: stats[feature]["std"] for feature in FEATURE_COLUMNS},
        feature_order=list(FEATURE_COLUMNS),
    )

    metrics = evaluate_model(model, test_records)
    report = generate_risk_report(model, records)
    summary = summarise_risk_levels(report)

    return TrainingArtifacts(model=model, metrics=metrics, risk_report=report, risk_summary=summary)


def save_artifacts(artifacts: TrainingArtifacts, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "intercept": artifacts.model.intercept,
        "coefficients": artifacts.model.coefficients,
        "feature_means": artifacts.model.feature_means,
        "feature_stds": artifacts.model.feature_stds,
        "feature_order": artifacts.model.feature_order,
    }
    with (directory / "risk_model.json").open("w", encoding="utf-8") as fp:
        json.dump(model_payload, fp, ensure_ascii=False, indent=2)

    with (directory / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(artifacts.metrics, fp, ensure_ascii=False, indent=2)

    report_path = directory / "risk_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(artifacts.risk_report[0].keys()))
        writer.writeheader()
        for row in artifacts.risk_report:
            writer.writerow(row)

    summary_path = directory / "risk_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(artifacts.risk_summary[0].keys()))
        writer.writeheader()
        for row in artifacts.risk_summary:
            writer.writerow(row)

