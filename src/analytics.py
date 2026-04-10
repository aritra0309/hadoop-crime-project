import json
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Ensure src imports work when launched via spark-submit.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.state_mapping import CANONICAL_TO_GEOJSON, get_geojson_name
except ModuleNotFoundError:
    from state_mapping import CANONICAL_TO_GEOJSON, get_geojson_name

try:
    from src.utils import safe_select
except ModuleNotFoundError:
    from utils import safe_select


HDFS_BASE = "hdfs://localhost:9000"
DISTRICT_PATH = f"{HDFS_BASE}/crime/output/district_master"
STATE_PATH = f"{HDFS_BASE}/crime/output/state_master"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "dashboard_data")


def canonical_to_geojson_name(state: Optional[str]) -> Optional[str]:
    if state is None:
        return None
    if state in CANONICAL_TO_GEOJSON:
        return CANONICAL_TO_GEOJSON[state]
    return get_geojson_name(state)


def hdfs_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hadoop_conf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def pick_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def existing(columns: Sequence[str], names: Sequence[str]) -> List[str]:
    col_set = {c.lower(): c for c in columns}
    return [col_set[n.lower()] for n in names if n.lower() in col_set]


def list_prefix(columns: Sequence[str], prefix: str, exclude: Optional[Sequence[str]] = None) -> List[str]:
    ex = set(x.lower() for x in (exclude or []))
    out = []
    for c in columns:
        cl = c.lower()
        if cl.startswith(prefix.lower()) and cl not in ex:
            out.append(c)
    return out


def safe_ratio(numer: Any, denom: Any) -> Optional[float]:
    if numer is None or denom is None:
        return None
    try:
        n = float(numer)
        d = float(denom)
    except (TypeError, ValueError):
        return None
    if d == 0 or math.isinf(d) or math.isnan(d):
        return None
    val = n / d
    if math.isinf(val) or math.isnan(val):
        return None
    return val


def round_float(x: Any, ndigits: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, ndigits)


def to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return to_native(obj.item())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray,)):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 2)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 2)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is None:
        return None
    return obj


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_native(payload), f, indent=2, ensure_ascii=True)


def rows_to_dicts(df: DataFrame, order_by: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if order_by:
        df = df.orderBy(*order_by)
    return [r.asDict(recursive=True) for r in df.collect()]


def with_min_max_scaled(df: DataFrame, value_col: str, out_col: str, partition_cols: Sequence[str]) -> DataFrame:
    w = Window.partitionBy(*partition_cols)
    min_c = F.min(F.col(value_col)).over(w)
    max_c = F.max(F.col(value_col)).over(w)
    return df.withColumn(
        out_col,
        F.when(
            max_c > min_c,
            ((F.col(value_col) - min_c) / (max_c - min_c)) * F.lit(100.0),
        ).otherwise(F.lit(0.0)),
    )


def non_empty_payload(payload: Dict[str, Any]) -> bool:
    for v in payload.values():
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, dict) and len(v) > 0:
            return True
    return False


def compute_district_analysis(district_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = district_df.columns

    total_ipc_col = pick_col(cols, ["total_ipc", "total_ipc_crimes"])
    total_women_col = pick_col(cols, ["total_women", "total_crimes_women"])

    severity_weights = {
        "murder": 10,
        "rape": 8,
        "kidnapping": 6,
        "dacoity": 5,
        "robbery": 5,
        "burglary": 2,
        "theft": 1,
        "riots": 3,
        "arson": 4,
        "dowry_deaths": 8,
    }
    sev_terms = [(c, w) for c, w in severity_weights.items() if c in cols]
    missing_sev_terms = [c for c in severity_weights if c not in cols]
    if missing_sev_terms:
        notes.append(f"District severity formula excluded missing columns: {', '.join(sorted(missing_sev_terms))}.")

    if not sev_terms:
        severity_df = district_df.select("state", "district", "year").withColumn("severity_score", F.lit(None))
    else:
        expr = None
        for c, w in sev_terms:
            term = F.coalesce(F.col(c), F.lit(0.0)) * F.lit(float(w))
            expr = term if expr is None else expr + term
        severity_df = district_df.withColumn("severity_raw", expr).select("state", "district", "year", "severity_raw")
        severity_df = with_min_max_scaled(severity_df, "severity_raw", "severity_score", ["year"])

    women_formula_candidates = [
        ("rape_women", 1.0),
        ("dowry_deaths_women", 2.0),
        ("domestic_cruelty", 1.0),
        ("assault_women", 1.0),
        ("kidnapping_women", 1.0),
    ]
    women_terms = [(c, w) for c, w in women_formula_candidates if c in cols]
    missing_women_terms = [c for c, _ in women_formula_candidates if c not in cols]
    if missing_women_terms:
        notes.append(f"District women safety formula excluded missing columns: {', '.join(sorted(missing_women_terms))}.")

    if not women_terms:
        women_df = district_df.select("state", "district", "year").withColumn("women_safety_index", F.lit(None))
    else:
        wexpr = None
        for c, w in women_terms:
            term = F.coalesce(F.col(c), F.lit(0.0)) * F.lit(float(w))
            wexpr = term if wexpr is None else wexpr + term
        women_df = district_df.withColumn("women_risk_raw", wexpr).select("state", "district", "year", "women_risk_raw")
        women_df = with_min_max_scaled(women_df, "women_risk_raw", "women_risk_norm", ["year"])
        women_df = women_df.withColumn("women_safety_index", F.lit(100.0) - F.col("women_risk_norm"))

    if total_ipc_col:
        yoy_w = Window.partitionBy("state", "district").orderBy("year")
        yoy_df = district_df.select("state", "district", "year", F.col(total_ipc_col).alias("total_ipc"))
        yoy_df = yoy_df.withColumn("prev_total_ipc", F.lag("total_ipc").over(yoy_w))
        yoy_df = yoy_df.withColumn(
            "yoy_growth_pct",
            F.when(
                (F.col("prev_total_ipc").isNull()) | (F.col("prev_total_ipc") == 0),
                F.lit(None),
            ).otherwise(((F.col("total_ipc") - F.col("prev_total_ipc")) / F.col("prev_total_ipc")) * F.lit(100.0)),
        )
    else:
        notes.append("District YoY growth could not be computed because total IPC column was not found.")
        yoy_df = district_df.select("state", "district", "year").withColumn("total_ipc", F.lit(None)).withColumn("prev_total_ipc", F.lit(None)).withColumn("yoy_growth_pct", F.lit(None))

    hotspots_2014_df = (
        severity_df.filter(F.col("year") == 2014)
        .select("state", "district", "year", "severity_score")
        .orderBy(F.col("severity_score").desc_nulls_last())
        .limit(50)
    )

    rising_df = (
        yoy_df.filter((F.col("year") >= 2010) & (F.col("year") <= 2014))
        .groupBy("state", "district")
        .agg(F.avg("yoy_growth_pct").alias("avg_yoy_growth_2010_2014"))
        .orderBy(F.col("avg_yoy_growth_2010_2014").desc_nulls_last())
        .limit(50)
    )

    if total_ipc_col and total_women_col and ("total_violent" in cols) and ("total_property" in cols):
        prof_df = district_df.select(
            "state",
            "district",
            "year",
            F.col(total_ipc_col).alias("total_ipc"),
            F.col("total_violent").alias("total_violent"),
            F.col("total_property").alias("total_property"),
            F.col(total_women_col).alias("total_women"),
        )
        prof_df = prof_df.withColumn(
            "pct_violent",
            F.when(F.col("total_ipc") > 0, F.col("total_violent") / F.col("total_ipc") * 100.0).otherwise(F.lit(None)),
        ).withColumn(
            "pct_property",
            F.when(F.col("total_ipc") > 0, F.col("total_property") / F.col("total_ipc") * 100.0).otherwise(F.lit(None)),
        ).withColumn(
            "pct_women",
            F.when(F.col("total_ipc") > 0, F.col("total_women") / F.col("total_ipc") * 100.0).otherwise(F.lit(None)),
        )
        prof_df = prof_df.withColumn(
            "dominant_crime_type",
            F.when((F.col("pct_violent") >= F.col("pct_property")) & (F.col("pct_violent") >= F.col("pct_women")), F.lit("violent"))
            .when((F.col("pct_property") >= F.col("pct_violent")) & (F.col("pct_property") >= F.col("pct_women")), F.lit("property"))
            .otherwise(F.lit("women")),
        )
    else:
        notes.append("District crime profiles used available columns only; one or more of total_violent/total_property/total_women/total_ipc was missing.")
        prof_df = district_df.select("state", "district", "year").withColumn("pct_violent", F.lit(None)).withColumn("pct_property", F.lit(None)).withColumn("pct_women", F.lit(None)).withColumn("dominant_crime_type", F.lit(None))

    payload = {
        "metadata": {
            "total_ipc_column": total_ipc_col,
            "total_women_column": total_women_col,
            "severity_terms_used": [{"column": c, "weight": w} for c, w in sev_terms],
            "women_safety_terms_used": [{"column": c, "weight": w} for c, w in women_terms],
            "deviations": notes,
        },
        "severity_scores": rows_to_dicts(severity_df.select("state", "district", "year", "severity_score"), ["state", "district", "year"]),
        "women_safety_index": rows_to_dicts(women_df.select("state", "district", "year", "women_safety_index"), ["state", "district", "year"]),
        "yoy_growth": rows_to_dicts(yoy_df.select("state", "district", "year", "total_ipc", "prev_total_ipc", "yoy_growth_pct"), ["state", "district", "year"]),
        "hotspots_2014": rows_to_dicts(hotspots_2014_df, ["severity_score"]),
        "rising_hotspots": rows_to_dicts(rising_df, ["avg_yoy_growth_2010_2014"]),
        "crime_profiles": rows_to_dicts(prof_df.select("state", "district", "year", "pct_violent", "pct_property", "pct_women", "dominant_crime_type"), ["state", "district", "year"]),
    }
    return payload


def compute_clusters(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns
    total_ipc_col = pick_col(cols, ["total_ipc", "total_ipc_crimes"])
    total_women_col = pick_col(cols, ["total_women", "total_crimes_women"])

    feature_candidates = {
        "avg_ipc_crimes": total_ipc_col,
        "avg_crimes_women": total_women_col,
        "avg_murder": "murder" if "murder" in cols else None,
        "avg_rape": "rape" if "rape" in cols else None,
        "avg_robbery": "robbery" if "robbery" in cols else None,
        "avg_recovery_rate": "recovery_rate" if "recovery_rate" in cols else None,
        "avg_auto_theft": pick_col(cols, ["auto_theft_stolen_state", "auto_theft_stolen_total", "auto_theft"]),
        "avg_fraud": pick_col(cols, ["fraud_total_bracket_sum", "total_fraud_cases", "fraud_total"]),
        "avg_firearms_murder": pick_col(cols, ["murder_by_firearms"]),
    }

    aggs = []
    if total_ipc_col:
        aggs.append(F.avg(total_ipc_col).alias("avg_ipc_crimes"))
        aggs.append(F.stddev(total_ipc_col).alias("ipc_variability"))
    else:
        notes.append("State clustering missing total IPC column; ipc features degraded.")

    for feat_name, src_col in feature_candidates.items():
        if feat_name == "avg_ipc_crimes":
            continue
        if src_col and src_col in cols:
            aggs.append(F.avg(src_col).alias(feat_name))
        elif feat_name != "avg_firearms_murder":
            notes.append(f"State clustering feature {feat_name} unavailable (column not found).")

    if not aggs:
        return {
            "optimal_k": None,
            "silhouette_scores": [],
            "state_assignments": [],
            "cluster_summaries": [],
            "labeling_metadata": {"deviations": notes},
        }

    feats_df = state_df.groupBy("state").agg(*aggs).na.fill(0.0)
    feature_cols = [c for c in feats_df.columns if c not in {"state"}]

    if len(feature_cols) < 2:
        notes.append("State clustering needs at least 2 features; produced fallback single cluster.")
        assigns = rows_to_dicts(feats_df.select("state").withColumn("cluster_id", F.lit(0)).withColumn("cluster_label", F.lit("Moderate Crime")), ["state"])
        for a in assigns:
            a["geojson_state"] = canonical_to_geojson_name(a["state"])
        return {
            "optimal_k": 1,
            "silhouette_scores": [],
            "state_assignments": assigns,
            "cluster_summaries": [{"cluster_id": 0, "cluster_label": "Moderate Crime"}],
            "labeling_metadata": {"deviations": notes},
        }

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled = assembler.transform(feats_df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaled = scaler.fit(assembled).transform(assembled)

    n_states = scaled.count()
    k_max = min(8, max(2, n_states - 1))

    evaluator = ClusteringEvaluator(featuresCol="scaled_features", predictionCol="prediction", metricName="silhouette")
    silhouettes = []
    for k in range(2, k_max + 1):
        model = KMeans(k=k, seed=42, featuresCol="scaled_features", predictionCol="prediction").fit(scaled)
        pred = model.transform(scaled)
        score = evaluator.evaluate(pred)
        silhouettes.append({"k": k, "score": score})

    best = max(silhouettes, key=lambda x: x["score"]) if silhouettes else {"k": 2, "score": 0.0}
    optimal_k = int(best["k"])

    model = KMeans(k=optimal_k, seed=42, featuresCol="scaled_features", predictionCol="cluster_id").fit(scaled)
    clustered = model.transform(scaled)

    global_stats = feats_df.select(
        *([F.mean(c).alias(f"{c}__mean") for c in feature_cols] + [F.stddev(c).alias(f"{c}__std") for c in feature_cols])
    ).collect()[0].asDict()

    cluster_means_df = clustered.groupBy("cluster_id").agg(*[F.mean(c).alias(c) for c in feature_cols], F.count("state").alias("state_count"))
    cluster_means = rows_to_dicts(cluster_means_df, ["cluster_id"])

    label_map: Dict[int, str] = {}
    secondary_threshold_z = 0.75
    for row in cluster_means:
        cluster_id = int(row["cluster_id"])
        ipc_mean = float(row.get("avg_ipc_crimes") or 0.0)
        g_mean = float(global_stats.get("avg_ipc_crimes__mean") or 0.0)
        g_std = float(global_stats.get("avg_ipc_crimes__std") or 0.0)

        if g_std > 0 and ipc_mean > g_mean + g_std:
            prefix = "High Crime"
        elif g_std > 0 and ipc_mean < g_mean - g_std:
            prefix = "Low Crime"
        else:
            prefix = "Moderate Crime"

        qualifier = None
        secondary_priority = [
            ("avg_recovery_rate", "Efficient Recovery", "Low Recovery"),
            ("avg_crimes_women", "High Women Crime", None),
            ("avg_fraud", "High Fraud", None),
            ("avg_firearms_murder", "High Firearms Use", None),
        ]
        best_abs_z = 0.0
        for feat, pos_label, neg_label in secondary_priority:
            if feat not in row:
                continue
            c_mean = float(row.get(feat) or 0.0)
            gg_mean = float(global_stats.get(f"{feat}__mean") or 0.0)
            gg_std = float(global_stats.get(f"{feat}__std") or 0.0)
            if gg_std <= 0:
                continue
            z = (c_mean - gg_mean) / gg_std
            if abs(z) > best_abs_z and abs(z) >= secondary_threshold_z:
                best_abs_z = abs(z)
                if z > 0:
                    qualifier = pos_label
                elif neg_label:
                    qualifier = neg_label

        label_map[cluster_id] = f"{prefix}, {qualifier}" if qualifier else prefix

    assigns_df = clustered.select("state", "cluster_id")
    assigns = rows_to_dicts(assigns_df, ["state"])
    for r in assigns:
        cid = int(r["cluster_id"])
        r["cluster_label"] = label_map.get(cid, "Moderate Crime")
        r["geojson_state"] = canonical_to_geojson_name(r["state"])

    summaries = []
    for row in cluster_means:
        cid = int(row["cluster_id"])
        out = {
            "cluster_id": cid,
            "cluster_label": label_map.get(cid, "Moderate Crime"),
            "state_count": row.get("state_count"),
        }
        for c in feature_cols:
            out[c] = row.get(c)
        summaries.append(out)

    return {
        "optimal_k": optimal_k,
        "silhouette_scores": silhouettes,
        "state_assignments": assigns,
        "cluster_summaries": summaries,
        "labeling_metadata": {
            "primary_threshold": "avg_ipc_crimes compared to global mean +/- 1 std",
            "secondary_threshold": f"largest |z| secondary feature with |z| >= {secondary_threshold_z}",
            "features_used": feature_cols,
            "deviations": notes,
        },
    }


def compute_crime_profiles(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns
    total_ipc_col = pick_col(cols, ["total_ipc", "total_ipc_crimes"])
    if not total_ipc_col:
        notes.append("Crime composition unavailable: total IPC column missing.")
        return {"metadata": {"deviations": notes}, "state_composition": [], "radar_dimensions": []}

    base_crime_cols = existing(
        cols,
        ["murder", "rape", "kidnapping", "robbery", "burglary", "theft", "riots", "cheating", "arson", "dowry_deaths"],
    )

    agg_exprs = [F.sum(total_ipc_col).alias("total_ipc_sum")]
    agg_exprs.extend([F.sum(c).alias(c) for c in base_crime_cols])
    by_state = state_df.groupBy("state").agg(*agg_exprs)

    detailed_rows = []
    radar_rows = []

    fraud_col = pick_col(cols, ["fraud_total_bracket_sum", "total_fraud_cases", "fraud_total"])
    breach_col = pick_col(cols, ["criminal_breach_of_trust"])
    auto_col = pick_col(cols, ["auto_theft", "auto_theft_stolen_state", "auto_theft_stolen_total"])

    for r in by_state.collect():
        d = r.asDict()
        state = d["state"]
        total = d.get("total_ipc_sum")
        out = {"state": state, "geojson_state": canonical_to_geojson_name(state)}
        for c in base_crime_cols:
            out[c] = (safe_ratio(d.get(c), total) or 0.0) * 100.0 if total else None
        detailed_rows.append(out)

        def sum_vals(names: Sequence[str]) -> float:
            total_local = 0.0
            for n in names:
                if n in cols:
                    q = state_df.filter(F.col("state") == state).agg(F.sum(n).alias("v")).collect()[0]["v"]
                    total_local += float(q or 0.0)
            return total_local

        violent_components = [c for c in ["murder", "rape", "kidnapping", "dacoity"] if c in cols]
        property_components = [c for c in ["burglary", "theft", "auto_theft", "cheating"] if c in cols]
        women_components = [c for c in ["dowry_deaths", "assault_women", "domestic_cruelty"] if c in cols]
        public_order_components = [c for c in ["riots", "arson"] if c in cols]
        white_collar_components = [c for c in ["cheating", breach_col, fraud_col] if c and c in cols]

        violent_val = sum_vals(violent_components)
        property_val = sum_vals(property_components)
        women_val = sum_vals(women_components)
        public_val = sum_vals(public_order_components)
        white_val = sum_vals(white_collar_components)

        radar_rows.append(
            {
                "state": state,
                "geojson_state": canonical_to_geojson_name(state),
                "Violent": (safe_ratio(violent_val, total) or 0.0) * 100.0 if total else None,
                "Property": (safe_ratio(property_val, total) or 0.0) * 100.0 if total else None,
                "Women": (safe_ratio(women_val, total) or 0.0) * 100.0 if total else None,
                "Public Order": (safe_ratio(public_val, total) or 0.0) * 100.0 if total else None,
                "White Collar": (safe_ratio(white_val, total) or 0.0) * 100.0 if total else None,
            }
        )

    return {
        "metadata": {
            "total_ipc_column": total_ipc_col,
            "detailed_crime_columns": base_crime_cols,
            "radar_grouping": {
                "Violent": [c for c in ["murder", "rape", "kidnapping", "dacoity"] if c in cols],
                "Property": [c for c in ["burglary", "theft", "auto_theft", "cheating"] if c in cols],
                "Women": [c for c in ["dowry_deaths", "assault_women", "domestic_cruelty"] if c in cols],
                "Public Order": [c for c in ["riots", "arson"] if c in cols],
                "White Collar": [c for c in ["cheating", breach_col, fraud_col] if c and c in cols],
            },
            "deviations": notes,
        },
        "state_composition": detailed_rows,
        "radar_dimensions": radar_rows,
    }


def compute_women_safety(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns
    total_women_col = pick_col(cols, ["total_women", "total_crimes_women"])

    women_terms = [(c, w) for c, w in [
        ("rape_women", 1.0),
        ("dowry_deaths_women", 2.0),
        ("domestic_cruelty", 1.0),
        ("assault_women", 1.0),
        ("kidnapping_women", 1.0),
    ] if c in cols]

    if women_terms:
        expr = None
        for c, w in women_terms:
            term = F.coalesce(F.col(c), F.lit(0.0)) * F.lit(float(w))
            expr = term if expr is None else expr + term
        risk_df = state_df.withColumn("women_risk_raw", expr).select("state", "year", "women_risk_raw")
        risk_df = with_min_max_scaled(risk_df, "women_risk_raw", "women_risk_norm", ["year"])
        safety_df = risk_df.withColumn("women_safety_index", F.lit(100.0) - F.col("women_risk_norm"))
    else:
        notes.append("Women safety index formula terms were not available.")
        safety_df = state_df.select("state", "year").withColumn("women_safety_index", F.lit(None))

    reported_col = pick_col(cols, ["women_cases_reported", "cases_reported"])
    chargesheet_col = pick_col(cols, ["women_cases_chargesheeted", "cases_chargesheeted"])
    convicted_col = pick_col(cols, ["women_cases_convicted", "cases_convicted"])
    trial_col = pick_col(cols, ["women_cases_sent_for_trial", "cases_sent_for_trial"])

    jp = []
    if reported_col and chargesheet_col and convicted_col and trial_col:
        jp_df = state_df.groupBy("state").agg(
            F.avg(reported_col).alias("avg_reported"),
            F.avg(chargesheet_col).alias("avg_chargesheeted"),
            F.avg(convicted_col).alias("avg_convicted"),
            F.avg(trial_col).alias("avg_sent_for_trial"),
        )
        for r in jp_df.collect():
            d = r.asDict()
            d["chargesheet_rate"] = (safe_ratio(d.get("avg_chargesheeted"), d.get("avg_reported")) or 0.0) if d.get("avg_reported") else None
            d["conviction_rate"] = (safe_ratio(d.get("avg_convicted"), d.get("avg_sent_for_trial")) or 0.0) if d.get("avg_sent_for_trial") else None
            d["geojson_state"] = canonical_to_geojson_name(d["state"])
            jp.append(d)
    else:
        notes.append("Women justice pipeline columns were partially unavailable.")

    breakdown_cols = [c for c in ["rape_women", "dowry_deaths_women", "kidnapping_women", "domestic_cruelty", "assault_women"] if c in cols]
    break_rows = []
    if total_women_col and breakdown_cols:
        agg = state_df.groupBy("state").agg(
            F.sum(total_women_col).alias("total_women_sum"),
            *[F.sum(c).alias(c) for c in breakdown_cols],
        )
        for r in agg.collect():
            d = r.asDict()
            total = d.get("total_women_sum")
            row = {"state": d["state"], "geojson_state": canonical_to_geojson_name(d["state"])}
            for c in breakdown_cols:
                row[c] = (safe_ratio(d.get(c), total) or 0.0) * 100.0 if total else None
            break_rows.append(row)

    nat_cols = [c for c in ["rape_women", "dowry_deaths_women", "kidnapping_women", "domestic_cruelty", "assault_women"] if c in cols]
    nat_trend = []
    if nat_cols:
        nat_df = state_df.groupBy("year").agg(
            *[F.sum(c).alias(c) for c in nat_cols],
            F.sum(total_women_col).alias("total_women") if total_women_col else F.lit(None).alias("total_women"),
        ).orderBy("year")
        nat_trend = rows_to_dicts(nat_df, ["year"])

    return {
        "metadata": {
            "women_terms_used": [{"column": c, "weight": w} for c, w in women_terms],
            "justice_pipeline_columns": {
                "reported": reported_col,
                "chargesheeted": chargesheet_col,
                "convicted": convicted_col,
                "sent_for_trial": trial_col,
            },
            "deviations": notes,
        },
        "state_safety_index": rows_to_dicts(safety_df.select("state", "year", "women_safety_index"), ["state", "year"]),
        "justice_pipeline": jp,
        "crime_type_breakdown": break_rows,
        "national_trend": nat_trend,
    }


def compute_supplementary(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns

    property_stolen_col = pick_col(cols, ["value_stolen"])
    property_recovered_col = pick_col(cols, ["value_recovered"])
    recovery_rate_col = pick_col(cols, ["recovery_rate"])

    prop_state = []
    prop_trend = []
    if property_stolen_col and property_recovered_col:
        p_state_df = state_df.groupBy("state").agg(
            F.sum(property_stolen_col).alias("total_stolen"),
            F.sum(property_recovered_col).alias("total_recovered"),
            F.avg(recovery_rate_col).alias("avg_recovery_rate") if recovery_rate_col else F.lit(None).alias("avg_recovery_rate"),
        )
        for r in p_state_df.collect():
            d = r.asDict()
            d["recovery_rate_computed"] = safe_ratio(d.get("total_recovered"), d.get("total_stolen"))
            d["geojson_state"] = canonical_to_geojson_name(d["state"])
            prop_state.append(d)

        p_trend_df = state_df.groupBy("year").agg(
            F.sum(property_stolen_col).alias("total_stolen"),
            F.sum(property_recovered_col).alias("total_recovered"),
        ).orderBy("year")
        for r in p_trend_df.collect():
            d = r.asDict()
            d["recovery_rate"] = safe_ratio(d.get("total_recovered"), d.get("total_stolen"))
            prop_trend.append(d)
    else:
        notes.append("Property recovery section partially unavailable (value_stolen/value_recovered not found).")

    kidnap_cols = [c for c in cols if c.startswith("kidnap_cases_") and c not in {"kidnap_cases_total"}]
    kidnap_state = []
    kidnap_trend = []
    if kidnap_cols:
        k_state_df = state_df.groupBy("state").agg(*[F.sum(c).alias(c) for c in kidnap_cols])
        for r in k_state_df.collect():
            d = r.asDict()
            total = sum(float(d.get(c) or 0.0) for c in kidnap_cols)
            row = {"state": d["state"], "geojson_state": canonical_to_geojson_name(d["state"])}
            for c in kidnap_cols:
                row[c] = (safe_ratio(d.get(c), total) or 0.0) * 100.0 if total else None
            kidnap_state.append(row)

        k_trend_df = state_df.groupBy("year").agg(*[F.sum(c).alias(c) for c in kidnap_cols]).orderBy("year")
        for r in k_trend_df.collect():
            d = r.asDict()
            total = sum(float(d.get(c) or 0.0) for c in kidnap_cols)
            row = {"year": d["year"]}
            for c in kidnap_cols:
                row[c] = (safe_ratio(d.get(c), total) or 0.0) * 100.0 if total else None
            kidnap_trend.append(row)

    murder_male = pick_col(cols, ["murder_victims_male"])
    murder_female = pick_col(cols, ["murder_victims_female"])
    murder_total = pick_col(cols, ["murder_victims_total"])
    murder_age_cols = existing(cols, ["murder_age_upto_10", "murder_age_10_15", "murder_age_15_18", "murder_age_18_30", "murder_age_30_50", "murder_age_above_50"])

    demo_state = []
    demo_trend = []
    if murder_male and murder_female:
        m_state_df = state_df.groupBy("state").agg(
            F.sum(murder_male).alias("male"),
            F.sum(murder_female).alias("female"),
            F.sum(murder_total).alias("total") if murder_total else (F.sum(murder_male) + F.sum(murder_female)).alias("total"),
            *[F.sum(c).alias(c) for c in murder_age_cols],
        )
        for r in m_state_df.collect():
            d = r.asDict()
            total = d.get("total")
            row = {
                "state": d["state"],
                "geojson_state": canonical_to_geojson_name(d["state"]),
                "male_pct": (safe_ratio(d.get("male"), total) or 0.0) * 100.0 if total else None,
                "female_pct": (safe_ratio(d.get("female"), total) or 0.0) * 100.0 if total else None,
            }
            age_total = sum(float(d.get(c) or 0.0) for c in murder_age_cols)
            for c in murder_age_cols:
                row[c] = (safe_ratio(d.get(c), age_total) or 0.0) * 100.0 if age_total else None
            demo_state.append(row)

        m_trend_df = state_df.groupBy("year").agg(
            F.sum(murder_male).alias("male"),
            F.sum(murder_female).alias("female"),
            F.sum(murder_total).alias("total") if murder_total else (F.sum(murder_male) + F.sum(murder_female)).alias("total"),
        ).orderBy("year")
        for r in m_trend_df.collect():
            d = r.asDict()
            total = d.get("total")
            d["male_pct"] = (safe_ratio(d.get("male"), total) or 0.0) * 100.0 if total else None
            d["female_pct"] = (safe_ratio(d.get("female"), total) or 0.0) * 100.0 if total else None
            demo_trend.append(d)

    firearms = pick_col(cols, ["murder_by_firearms"])
    licensed = pick_col(cols, ["murder_by_licensed_arms"])
    unlicensed = pick_col(cols, ["murder_by_unlicensed_arms"])
    murder_col = pick_col(cols, ["murder"])

    firearm_state = []
    firearm_trend = []
    if firearms and murder_col:
        f_state_df = state_df.groupBy("state").agg(
            F.sum(firearms).alias("firearms_murders"),
            F.sum(murder_col).alias("total_murders"),
            F.sum(licensed).alias("licensed") if licensed else F.lit(None).alias("licensed"),
            F.sum(unlicensed).alias("unlicensed") if unlicensed else F.lit(None).alias("unlicensed"),
        )
        for r in f_state_df.collect():
            d = r.asDict()
            total = d.get("total_murders")
            arm_total = (d.get("licensed") or 0.0) + (d.get("unlicensed") or 0.0)
            d["firearms_pct_of_murders"] = (safe_ratio(d.get("firearms_murders"), total) or 0.0) * 100.0 if total else None
            d["licensed_share"] = (safe_ratio(d.get("licensed"), arm_total) or 0.0) * 100.0 if arm_total else None
            d["unlicensed_share"] = (safe_ratio(d.get("unlicensed"), arm_total) or 0.0) * 100.0 if arm_total else None
            d["geojson_state"] = canonical_to_geojson_name(d["state"])
            firearm_state.append(d)

        f_trend_df = state_df.groupBy("year").agg(
            F.sum(firearms).alias("firearms_murders"),
            F.sum(murder_col).alias("total_murders"),
            F.sum(licensed).alias("licensed") if licensed else F.lit(None).alias("licensed"),
            F.sum(unlicensed).alias("unlicensed") if unlicensed else F.lit(None).alias("unlicensed"),
        ).orderBy("year")
        for r in f_trend_df.collect():
            d = r.asDict()
            total = d.get("total_murders")
            arm_total = (d.get("licensed") or 0.0) + (d.get("unlicensed") or 0.0)
            d["firearms_pct_of_murders"] = (safe_ratio(d.get("firearms_murders"), total) or 0.0) * 100.0 if total else None
            d["licensed_share"] = (safe_ratio(d.get("licensed"), arm_total) or 0.0) * 100.0 if arm_total else None
            d["unlicensed_share"] = (safe_ratio(d.get("unlicensed"), arm_total) or 0.0) * 100.0 if arm_total else None
            firearm_trend.append(d)

    place_groups = {
        "residential": list_prefix(cols, "residential_"),
        "highway": list_prefix(cols, "highway_"),
        "railway": list_prefix(cols, "railway_"),
        "bank": list_prefix(cols, "bank_"),
        "commercial": list_prefix(cols, "commercial_"),
        "other_places": list_prefix(cols, "other_places_"),
        "river_sea": list_prefix(cols, "river_sea_"),
    }

    geo_state = []
    geo_trend = []
    if any(len(v) > 0 for v in place_groups.values()):
        state_sums = []
        for place, cols_list in place_groups.items():
            if cols_list:
                state_sums.append(F.sum(sum(F.coalesce(F.col(c), F.lit(0.0)) for c in cols_list)).alias(place))
        g_state_df = state_df.groupBy("state").agg(*state_sums)
        for r in g_state_df.collect():
            d = r.asDict()
            total = sum(float(d.get(p) or 0.0) for p in place_groups)
            row = {"state": d["state"], "geojson_state": canonical_to_geojson_name(d["state"])}
            for place in place_groups:
                row[place] = (safe_ratio(d.get(place), total) or 0.0) * 100.0 if total else None
            geo_state.append(row)

        trend_sums = []
        for place, cols_list in place_groups.items():
            if cols_list:
                trend_sums.append(F.sum(sum(F.coalesce(F.col(c), F.lit(0.0)) for c in cols_list)).alias(place))
        g_trend_df = state_df.groupBy("year").agg(*trend_sums).orderBy("year")
        for r in g_trend_df.collect():
            d = r.asDict()
            total = sum(float(d.get(p) or 0.0) for p in place_groups)
            row = {"year": d["year"]}
            for place in place_groups:
                row[place] = (safe_ratio(d.get(place), total) or 0.0) * 100.0 if total else None
            geo_trend.append(row)

    auto_stolen = pick_col(cols, ["auto_theft_stolen_state", "auto_theft_stolen_total", "auto_theft"])
    auto_recovered = pick_col(cols, ["auto_theft_recovered_state"])
    vehicle_cols = list_prefix(cols, "auto_theft_stolen_", exclude=[auto_stolen] if auto_stolen else [])

    auto_state = []
    auto_trend = []
    vehicle_breakdown = []
    if auto_stolen:
        a_state_df = state_df.groupBy("state").agg(
            F.sum(auto_stolen).alias("stolen"),
            F.sum(auto_recovered).alias("recovered") if auto_recovered else F.lit(None).alias("recovered"),
            *[F.sum(c).alias(c) for c in vehicle_cols],
        )
        for r in a_state_df.collect():
            d = r.asDict()
            d["recovery_rate"] = safe_ratio(d.get("recovered"), d.get("stolen"))
            d["geojson_state"] = canonical_to_geojson_name(d["state"])
            auto_state.append({k: d.get(k) for k in ["state", "geojson_state", "stolen", "recovered", "recovery_rate"]})
            if vehicle_cols:
                denom = d.get("stolen")
                vb = {"state": d["state"], "geojson_state": canonical_to_geojson_name(d["state"])}
                for c in vehicle_cols:
                    vb[c] = (safe_ratio(d.get(c), denom) or 0.0) * 100.0 if denom else None
                vehicle_breakdown.append(vb)

        a_trend_df = state_df.groupBy("year").agg(
            F.sum(auto_stolen).alias("stolen"),
            F.sum(auto_recovered).alias("recovered") if auto_recovered else F.lit(None).alias("recovered"),
        ).orderBy("year")
        for r in a_trend_df.collect():
            d = r.asDict()
            d["recovery_rate"] = safe_ratio(d.get("recovered"), d.get("stolen"))
            auto_trend.append(d)

    return {
        "metadata": {
            "supplementary_coverage_years": "2001-2010 expected; nulls in 2011-2014 treated as expected",
            "deviations": notes,
        },
        "property_recovery": {
            "state_level": prop_state,
            "national_trend": prop_trend,
        },
        "kidnapping_motives": {
            "state_level": kidnap_state,
            "national_trend": kidnap_trend,
        },
        "murder_demographics": {
            "state_level": demo_state,
            "national_trend": demo_trend,
        },
        "firearms": {
            "state_level": firearm_state,
            "national_trend": firearm_trend,
        },
        "crime_geography": {
            "state_level": geo_state,
            "national_trend": geo_trend,
        },
        "auto_theft": {
            "state_level": auto_state,
            "national_trend": auto_trend,
            "vehicle_type_mix": vehicle_breakdown,
        },
    }


def compute_forecasts(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns
    total_ipc_col = pick_col(cols, ["total_ipc", "total_ipc_crimes"])
    total_women_col = pick_col(cols, ["total_women", "total_crimes_women"])

    metrics = []
    if total_ipc_col:
        metrics.append(("total_ipc", total_ipc_col))
    if total_women_col:
        metrics.append(("total_women", total_women_col))
    if not metrics:
        notes.append("Forecasting skipped: neither IPC nor women total columns were available.")
        return {"metadata": {"deviations": notes}, "time_series": [], "model_metadata": []}

    pdf = state_df.select("state", "year", *[src for _, src in metrics]).toPandas()
    pdf = pdf.sort_values(["state", "year"]).reset_index(drop=True)

    time_series = []
    model_meta = []

    for state, grp in pdf.groupby("state"):
        for metric_out, metric_src in metrics:
            sub = grp[["year", metric_src]].dropna().sort_values("year")
            if sub.empty:
                notes.append(f"Forecasting skipped for state={state}, metric={metric_out}: no non-null observations.")
                continue

            years = sub["year"].to_numpy(dtype=float)
            y = sub[metric_src].to_numpy(dtype=float)
            x = (years - 2001.0).reshape(-1, 1)

            candidate_models = {
                "poly2_ridge_a10": Pipeline([
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("ridge", Ridge(alpha=10.0)),
                ]),
                "poly3_ridge_a50": Pipeline([
                    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                    ("ridge", Ridge(alpha=50.0)),
                ]),
            }

            best_name = None
            best_model = None
            best_mae = None

            if len(sub) >= 6:
                tscv = TimeSeriesSplit(n_splits=3)
                for name, model in candidate_models.items():
                    try:
                        scores = cross_val_score(model, x, y, cv=tscv, scoring="neg_mean_absolute_error")
                        mae = float(-scores.mean())
                        if best_mae is None or mae < best_mae:
                            best_mae = mae
                            best_name = name
                            best_model = model
                    except Exception:
                        continue
            else:
                notes.append(f"Forecasting CV degraded for state={state}, metric={metric_out}: <6 observations, using default poly2 model.")

            if best_model is None:
                best_name = "poly2_ridge_a10"
                best_model = candidate_models[best_name]
                best_mae = None

            best_model.fit(x, y)
            train_pred = best_model.predict(x)
            train_r2 = float(r2_score(y, train_pred)) if len(y) >= 2 else None

            for yr, val in zip(years, y):
                time_series.append(
                    {
                        "state": state,
                        "year": int(yr),
                        "value": float(val),
                        "metric": metric_out,
                        "type": "actual",
                    }
                )

            future_years = np.arange(2015, 2021, dtype=float)
            xf = (future_years - 2001.0).reshape(-1, 1)
            preds = best_model.predict(xf)
            preds = np.clip(preds, 0, None)

            for yr, val in zip(future_years, preds):
                time_series.append(
                    {
                        "state": state,
                        "year": int(yr),
                        "value": float(val),
                        "metric": metric_out,
                        "type": "predicted",
                    }
                )

            model_meta.append(
                {
                    "state": state,
                    "metric": metric_out,
                    "best_model": best_name,
                    "cv_mae": best_mae,
                    "train_r2": train_r2,
                }
            )

    time_series = sorted(time_series, key=lambda x: (x["state"], x["metric"], x["year"], x["type"]))
    model_meta = sorted(model_meta, key=lambda x: (x["state"], x["metric"]))

    return {
        "metadata": {
            "models": [
                "PolynomialFeatures(degree=2)+Ridge(alpha=10.0)",
                "PolynomialFeatures(degree=3)+Ridge(alpha=50.0)",
            ],
            "selection": "TimeSeriesSplit(n_splits=3) by neg_mean_absolute_error when sufficient samples",
            "input_feature": "t = year - 2001",
            "deviations": notes,
        },
        "time_series": time_series,
        "model_metadata": model_meta,
    }


def compute_national_trends(state_df: DataFrame, notes: List[str]) -> Dict[str, Any]:
    cols = state_df.columns
    total_ipc_col = pick_col(cols, ["total_ipc", "total_ipc_crimes"])
    total_women_col = pick_col(cols, ["total_women", "total_crimes_women"])

    if not total_ipc_col or not total_women_col:
        notes.append("National yearly totals degraded because total IPC or total women column was missing.")

    yearly = state_df.groupBy("year").agg(
        F.sum(total_ipc_col).alias("total_ipc") if total_ipc_col else F.lit(None).alias("total_ipc"),
        F.sum(total_women_col).alias("total_women") if total_women_col else F.lit(None).alias("total_women"),
    ).orderBy("year")
    yearly_rows = rows_to_dicts(yearly, ["year"])

    map_year = {r["year"]: r for r in yearly_rows}
    overall_change_pct = None
    if 2001 in map_year and 2014 in map_year and map_year[2001].get("total_ipc"):
        overall_change_pct = (safe_ratio(map_year[2014].get("total_ipc") - map_year[2001].get("total_ipc"), map_year[2001].get("total_ipc")) or 0.0) * 100.0

    candidate_crimes = existing(
        cols,
        [
            "murder", "rape", "kidnapping", "robbery", "burglary", "theft", "riots", "cheating", "arson", "dowry_deaths",
            "dacoity", "attempt_murder", "culpable_homicide", "auto_theft", "criminal_breach_of_trust", "hurt", "other_ipc",
        ],
    )

    changes = []
    if candidate_crimes:
        y2001 = state_df.filter(F.col("year") == 2001).agg(*[F.sum(c).alias(c) for c in candidate_crimes]).collect()[0].asDict()
        y2014 = state_df.filter(F.col("year") == 2014).agg(*[F.sum(c).alias(c) for c in candidate_crimes]).collect()[0].asDict()
        for c in candidate_crimes:
            b = y2001.get(c)
            e = y2014.get(c)
            if b is None or e is None or float(b) == 0.0:
                continue
            pct = ((float(e) - float(b)) / float(b)) * 100.0
            changes.append({"crime_type": c, "pct_change": pct, "value_2001": b, "value_2014": e})

    fastest_growing = sorted(changes, key=lambda x: x["pct_change"], reverse=True)[:5]
    fastest_declining = [x for x in sorted(changes, key=lambda x: x["pct_change"]) if x["pct_change"] < 0][:5]

    return {
        "metadata": {
            "total_ipc_column": total_ipc_col,
            "total_women_column": total_women_col,
            "crime_columns_considered": candidate_crimes,
            "deviations": notes,
        },
        "yearly_totals": yearly_rows,
        "overall_change_pct": overall_change_pct,
        "fastest_growing": fastest_growing,
        "fastest_declining": fastest_declining,
    }


def validate_json_outputs(output_paths: List[str]) -> Dict[str, Any]:
    checks = []
    for p in output_paths:
        ok = True
        msg = "ok"
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not non_empty_payload(data):
                ok = False
                msg = "parsed but appears empty"
        except Exception as e:
            ok = False
            msg = str(e)
        checks.append({"file": p, "ok": ok, "message": msg, "size_bytes": os.path.getsize(p) if os.path.exists(p) else 0})
    return {"checks": checks, "all_ok": all(x["ok"] for x in checks)}


def section_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for k, v in payload.items():
        if isinstance(v, list):
            out[k] = len(v)
        elif isinstance(v, dict):
            out[k] = len(v)
    return out


def main() -> None:
    spark = (
        SparkSession.builder.appName("Crime Analytics Phase 2")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("=" * 88)
    print("PHASE 2 ANALYTICS - INPUT VALIDATION")
    print("=" * 88)

    district_ok = hdfs_exists(spark, DISTRICT_PATH)
    state_ok = hdfs_exists(spark, STATE_PATH)

    if not district_ok or not state_ok:
        print("ERROR: Required input master tables are missing.")
        print(f"Expected: {DISTRICT_PATH}")
        print(f"Expected: {STATE_PATH}")
        spark.stop()
        sys.exit(1)

    district_df = spark.read.parquet(DISTRICT_PATH)
    state_df = spark.read.parquet(STATE_PATH)

    print("\nDISTRICT MASTER SCHEMA")
    district_df.printSchema()
    print("DISTRICT MASTER ROW COUNT:", district_df.count())
    district_df.show(5, truncate=False)

    print("\nSTATE MASTER SCHEMA")
    state_df.printSchema()
    print("STATE MASTER ROW COUNT:", state_df.count())
    state_df.show(5, truncate=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    district_notes: List[str] = []
    cluster_notes: List[str] = []
    profile_notes: List[str] = []
    women_notes: List[str] = []
    supp_notes: List[str] = []
    forecast_notes: List[str] = []
    national_notes: List[str] = []

    district_payload = compute_district_analysis(district_df, district_notes)
    cluster_payload = compute_clusters(state_df, cluster_notes)
    profile_payload = compute_crime_profiles(state_df, profile_notes)
    women_payload = compute_women_safety(state_df, women_notes)
    supplementary_payload = compute_supplementary(state_df, supp_notes)
    forecast_payload = compute_forecasts(state_df, forecast_notes)
    national_payload = compute_national_trends(state_df, national_notes)

    outputs = {
        "district_analysis.json": district_payload,
        "clusters.json": cluster_payload,
        "crime_profiles.json": profile_payload,
        "women_safety.json": women_payload,
        "supplementary.json": supplementary_payload,
        "forecasts.json": forecast_payload,
        "national_trends.json": national_payload,
    }

    output_paths = []
    print("\nWRITING JSON OUTPUTS")
    for fname, payload in outputs.items():
        path = os.path.join(OUTPUT_DIR, fname)
        write_json(path, payload)
        output_paths.append(path)
        print(f"Wrote {path}")

    validation = validate_json_outputs(output_paths)
    print("\nJSON VALIDATION")
    for c in validation["checks"]:
        print(f"- {os.path.basename(c['file'])}: ok={c['ok']} size={c['size_bytes']} msg={c['message']}")

    print("\nSECTION RECORD COUNTS")
    for fname, payload in outputs.items():
        print(f"{fname}: {section_counts(payload)}")

    sample_states_df = safe_select(
        state_df.select("state").distinct().orderBy("state").limit(5),
        ["state"],
    )
    sample_states = [r["state"] for r in sample_states_df.collect()]
    print("\nSAMPLE STATES:", sample_states)

    if not validation["all_ok"]:
        print("\nERROR: One or more JSON outputs failed validation.")
        spark.stop()
        sys.exit(2)

    print("\nAll outputs generated successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
