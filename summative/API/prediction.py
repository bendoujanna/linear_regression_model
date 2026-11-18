from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR / "baseline_train_data.csv"

app = FastAPI(
    title="Healthcare Cost Prediction API",
    description="API for predicting inpatient hospital costs based on clinical factors.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://linear-regression-model-ry4e.onrender.com",  # Render URL
        "http://localhost:8000",                    # Flutter web debug port
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,  # no cookies or auth sessions are used by this API
)

try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    model_pipeline = None
    print(f"Warning: Could not load model at startup. {e}")

AGE_MAP = {"0 to 17": 0, "18 to 29": 1, "30 to 49": 2, "50 to 69": 3, "70 or Older": 4}
MORTALITY_MAP = {"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}
ADMISSION_MAP = {
    "Elective": 0, "Urgent": 1, "Emergency": 2,
    "Newborn": 3, "Trauma": 4, "Not Available": 5,
}
ED_MAP = {"N": 0, "Y": 1}
MED_SURG_MAP = {"Medical": 0, "Surgical": 1, "Not Applicable": 0}
PAYMENT_MAP = {
    "Blue Cross/Blue Shield": 0, "Department of Corrections": 1,
    "Federal/State/Local/VA": 2, "Managed Care, Unspecified": 3,
    "Medicaid": 4, "Medicare": 5, "Miscellaneous/Other": 6,
    "Private Health Insurance": 7, "Self-Pay": 8, "Unknown": 9,
}

FEATURE_COLS = [
    "Length of Stay",
    "APR Severity of Illness Code",
    "APR MDC Code",
    "Age Group",
    "APR Risk of Mortality",
    "Type of Admission",
    "Emergency Department Indicator",
    "APR Medical Surgical Description",
    "Payment Typology 1",
]
TARGET_COL = "Total Costs"


class PatientAdmissionInput(BaseModel):
    length_of_stay: int = Field(..., ge=1, le=120, description="Total days in hospital (1-120)")
    apr_mdc_code: int = Field(..., ge=1, le=25, description="Major Diagnostic Category (1-25)")
    apr_severity_of_illness_code: int = Field(..., ge=1, le=4, description="1: Minor, 2: Moderate, 3: Major, 4: Extreme")
    apr_risk_of_mortality: int = Field(..., ge=1, le=4, description="1: Minor, 2: Moderate, 3: Major, 4: Extreme")
    apr_medical_surgical_description: int = Field(..., ge=0, le=1, description="0: Medical, 1: Surgical")
    age_group: int = Field(..., ge=0, le=4, description="0: 0-17, 1: 18-29, 2: 30-49, 3: 50-69, 4: 70+")
    type_of_admission: int = Field(..., ge=0, le=5, description="0: Elective, 1: Urgent, 2: Emergency, 3: Newborn, 4: Trauma, 5: Not Available")
    emergency_department_indicator: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    payment_typology_1: int = Field(..., ge=0, le=9, description="0: Blue Cross/Blue Shield ... 9: Unknown (see PAYMENT_MAP)")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Healthcare Cost Prediction API. Model is loaded and ready."}


@app.post("/predict")
def predict_cost(patient: PatientAdmissionInput):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    # Map incoming Pydantic fields to the exact DataFrame column names the model saw in training
    input_dict = {
        "Length of Stay": [patient.length_of_stay],
        "APR Severity of Illness Code": [patient.apr_severity_of_illness_code],
        "APR MDC Code": [patient.apr_mdc_code],
        "Age Group": [patient.age_group],
        "APR Risk of Mortality": [patient.apr_risk_of_mortality],
        "Type of Admission": [patient.type_of_admission],
        "Emergency Department Indicator": [patient.emergency_department_indicator],
        "APR Medical Surgical Description": [patient.apr_medical_surgical_description],
        "Payment Typology 1": [patient.payment_typology_1],
    }
    input_df = pd.DataFrame(input_dict)[FEATURE_COLS]  # enforce exact column order/set

    try:
        prediction = model_pipeline.predict(input_df)[0]
        return {
            "predicted_total_cost": round(float(prediction), 2),
            "currency": "USD",
            "model_used": "RandomForestRegressor",
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Feature validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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

    # Apply the same categorical encodings used during training
    if "Age Group" in df.columns:
        df["Age Group"] = df["Age Group"].map(AGE_MAP)
    if "APR Risk of Mortality" in df.columns:
        df["APR Risk of Mortality"] = df["APR Risk of Mortality"].map(MORTALITY_MAP)
    if "Type of Admission" in df.columns:
        df["Type of Admission"] = df["Type of Admission"].map(ADMISSION_MAP)
    if "Emergency Department Indicator" in df.columns:
        df["Emergency Department Indicator"] = df["Emergency Department Indicator"].map(ED_MAP)
    if "APR Medical Surgical Description" in df.columns:
        df["APR Medical Surgical Description"] = df["APR Medical Surgical Description"].map(MED_SURG_MAP)
    if "Payment Typology 1" in df.columns:
        df["Payment Typology 1"] = df["Payment Typology 1"].map(PAYMENT_MAP)

    return df.dropna(subset=FEATURE_COLS + [TARGET_COL])


@app.post("/retrain")
def retrain_model(file: UploadFile = File(...)):
    try:
        new_df = pd.read_csv(file.file, low_memory=False)

        missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in new_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded dataset is missing required columns: {missing_cols}",
            )

        new_df = _clean_and_encode(new_df)

        baseline_df = pd.read_csv(DATA_PATH)
        combined_df = pd.concat(
            [baseline_df[FEATURE_COLS + [TARGET_COL]], new_df[FEATURE_COLS + [TARGET_COL]]],
            ignore_index=True,
        )

        X = combined_df[FEATURE_COLS]
        y = combined_df[TARGET_COL]
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

        global model_pipeline
        model_pipeline = new_pipeline
        joblib.dump(new_pipeline, MODEL_PATH)
        combined_df.to_csv(DATA_PATH, index=False)  # grows the baseline for next retrain

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