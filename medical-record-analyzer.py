import pandas as pd
import logging

# ----------------------------------------
# ログ設定（実務向け）
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="medical_system.log",
    filemode="w"
)

logging.info("=== 医療データ解析ツール 起動 ===")

# ----------------------------------------
# 模擬電子カルテ・検査データ（実務風）
# patient_id: 患者ID
# age: 年齢
# systolic_bp: 収縮期血圧
# diastolic_bp: 拡張期血圧
# glucose: 血糖値
# ----------------------------------------
data = [
    {"patient_id": "P001", "age": 45, "systolic_bp": 130, "diastolic_bp": 80, "glucose": 95},
    {"patient_id": "P002", "age": 70, "systolic_bp": 160, "diastolic_bp": 95, "glucose": 180},
    {"patient_id": "P003", "age": 55, "systolic_bp": None, "diastolic_bp": 88, "glucose": 110},
    {"patient_id": "P004", "age": 62, "systolic_bp": 145, "diastolic_bp": None, "glucose": 200},
    {"patient_id": "P005", "age": 50, "systolic_bp": 120, "diastolic_bp": 78, "glucose": None},
]

df = pd.DataFrame(data)
logging.info("電子カルテ・検査データ読み込み完了")

# ----------------------------------------
# 前処理：欠損値補完・型チェック
# ----------------------------------------
def preprocess_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 数値項目の欠損値を中央値で補完（実務でよくある方法）
    numeric_cols = ["age", "systolic_bp", "diastolic_bp", "glucose"]
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            logging.warning(f"{col} に欠損値あり → 中央値 {median_value} で補完")
            df[col].fillna(median_value, inplace=True)

    # 型チェック（数値でないものが混ざっていないか）
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.error(f"{col} に数値以外の値が含まれています")
            raise ValueError(f"Invalid data type in column: {col}")

    return df

df_clean = preprocess_medical_data(df)
logging.info("前処理完了")

# ----------------------------------------
# 異常検知ロジック（簡易版）
# ・高血圧疑い
# ・高血糖疑い
# ----------------------------------------
def detect_abnormal_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 高血圧疑い（収縮期140以上 or 拡張期90以上）
    df["hypertension_flag"] = (
        (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
    )

    # 高血糖疑い（血糖値 140以上）
    df["hyperglycemia_flag"] = df["glucose"] >= 140

    return df

df_result = detect_abnormal_values(df_clean)
logging.info("異常値判定完了")

# ----------------------------------------
# 集計レポート出力（コンソール）
# ----------------------------------------
def print_report(df: pd.DataFrame) -> None:
    total_patients = len(df)
    hypertension_count = df["hypertension_flag"].sum()
    hyperglycemia_count = df["hyperglycemia_flag"].sum()

    print("=== 医療データ解析レポート ===")
    print(f"患者数: {total_patients}")
    print(f"高血圧疑い: {hypertension_count} 名")
    print(f"高血糖疑い: {hyperglycemia_count} 名")
    print()
    print("=== 患者別判定結果 ===")
    for _, row in df.iterrows():
        flags = []
        if row["hypertension_flag"]:
            flags.append("高血圧疑い")
        if row["hyperglycemia_flag"]:
            flags.append("高血糖疑い")
        flag_text = " / ".join(flags) if flags else "異常なし"
        print(f"患者ID: {row['patient_id']} | 判定: {flag_text}")

print_report(df_result)
logging.info("レポート出力完了")
logging.info("=== 医療データ解析ツール 終了 ===")
