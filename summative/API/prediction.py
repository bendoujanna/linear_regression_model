from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR / "baseline_train_data.csv"

app = FastAPI(
    title="Healthcare Cost Prediction API",
    description="API for predicting inpatient hospital costs based on clinical factors.",
    version="1.0.0"
)

# Allow only my deployed Render URL and local dev port to call this API using GET/POST and the Content-Type header
# Restrict wildcard (*) origins, other HTTP methods/headers, and credentials (since this API uses no logins or cookies) so random websites cannot call it from a user's browser
# The Flutter mobile app is not a browser, so CORS does not block it either way; these rules only control browser-based callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://linear-regression-model-ry4e.onrender.com",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

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
    "Length of Stay", "APR Severity of Illness Code", "APR MDC Code",
    "Age Group", "APR Risk of Mortality", "Type of Admission",
    "Emergency Department Indicator", "APR Medical Surgical Description",
    "Payment Typology 1",
]
TARGET_COL = "Total Costs"


class ModelState:
    pipeline = None


try:
    ModelState.pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    ModelState.pipeline = None
    print(f"Warning: Could not load model at startup. {e}")


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
    if ModelState.pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

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
    input_df = pd.DataFrame(input_dict)[FEATURE_COLS]

    try:
        prediction = ModelState.pipeline.predict(input_df)[0]
        return {
            "predicted_total_cost": round(float(prediction), 2),
            "currency": "USD",
            "model_used": "RandomForestRegressor",
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Feature validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


from retrain import router as retrain_router
app.include_router(retrain_router)