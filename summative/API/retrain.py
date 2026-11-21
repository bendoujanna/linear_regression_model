from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import prediction  # reuses prediction.py's maps, paths, FEATURE_COLS and ModelState

router = APIRouter()


def _clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the notebook's cleaning + encoding so uploaded raw data matches training format."""
    df = df.copy()

    for col in ["Total Costs", "Total Charges"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Length of Stay" in df.columns:
        df["Length of Stay"] = (
            df["Length of Stay"].astype(str).str.replace("120 +", "120", regex=False)
        )
        df["Length of Stay"] = pd.to_numeric(df["Length of Stay"], errors="coerce")

    df = df.dropna(subset=["Total Costs", "Length of Stay"]).copy()
    upper_cap = df["Total Costs"].quantile(0.99)
    df = df[df["Total Costs"] <= upper_cap].copy()

    if "Age Group" in df.columns:
        df["Age Group"] = df["Age Group"].map(prediction.AGE_MAP)
    if "APR Risk of Mortality" in df.columns:
        df["APR Risk of Mortality"] = df["APR Risk of Mortality"].map(prediction.MORTALITY_MAP)
    if "Type of Admission" in df.columns:
        df["Type of Admission"] = df["Type of Admission"].map(prediction.ADMISSION_MAP)
    if "Emergency Department Indicator" in df.columns:
        df["Emergency Department Indicator"] = df["Emergency Department Indicator"].map(prediction.ED_MAP)
    if "APR Medical Surgical Description" in df.columns:
        df["APR Medical Surgical Description"] = df["APR Medical Surgical Description"].map(prediction.MED_SURG_MAP)
    if "Payment Typology 1" in df.columns:
        df["Payment Typology 1"] = df["Payment Typology 1"].map(prediction.PAYMENT_MAP)

    return df.dropna(subset=prediction.FEATURE_COLS + [prediction.TARGET_COL])


@router.post("/retrain")
def retrain_model(file: UploadFile = File(...)):
    try:
        new_df = pd.read_csv(file.file, low_memory=False)

        missing_cols = [c for c in prediction.FEATURE_COLS + [prediction.TARGET_COL] if c not in new_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded dataset is missing required columns: {missing_cols}",
            )

        new_df = _clean_and_encode(new_df)

        baseline_df = pd.read_csv(prediction.DATA_PATH)
        combined_df = pd.concat(
            [baseline_df[prediction.FEATURE_COLS + [prediction.TARGET_COL]],
             new_df[prediction.FEATURE_COLS + [prediction.TARGET_COL]]],
            ignore_index=True,
        )

        X = combined_df[prediction.FEATURE_COLS]
        y = combined_df[prediction.TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        new_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=8, random_state=42, n_jobs=-1
            )),
        ])
        new_pipeline.fit(X_train, y_train)

        test_preds = new_pipeline.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        prediction.ModelState.pipeline = new_pipeline
        joblib.dump(new_pipeline, prediction.MODEL_PATH)
        combined_df.to_csv(prediction.DATA_PATH, index=False)

        return {
            "message": "Model retrained and updated successfully!",
            "rows_added": len(new_df),
            "total_rows_used": len(combined_df),
            "new_test_r2": round(float(test_r2), 3),
            "new_test_rmse": round(float(test_rmse), 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining process failed: {str(e)}")