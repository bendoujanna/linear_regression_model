# Inpatient Hospital Cost Estimator

## Mission & Problem
My mission is to make healthcare more affordable and transparent for patients and their families. Many people only find out how much a hospital stay will cost after discharge, when it is too late to plan financially. This project predicts the total cost of an inpatient stay at admission using clinical and visit details, served through a public API and a Flutter mobile app.

## Dataset
This project uses the **2015 New York State SPARCS De-Identified Hospital Inpatient Discharges** dataset, published by the New York State Department of Health:
- **Source:** [Health Data NY — Hospital Inpatient Discharges (SPARCS De-Identified): 2015](https://www.kaggle.com/datasets/jonasalmeida/2015-deidentified-ny-inpatient-discharge-sparcs?
resource=download)

The dataset contains real inpatient discharge records across New York State with a mix of numeric and categorical columns. After sampling (~10,000 rows), cleaning currency strings, and capping the top 1% of extreme billing outliers, the model trains on 9 features available at admission (`Length of Stay`, `APR MDC Code`, `APR Severity of Illness Code`, `APR Risk of Mortality`, `APR Medical Surgical Description`, `Age Group`, `Type of Admission`, `Emergency Department Indicator`, and `Payment Typology 1`) to predict `Total Costs` (USD).

## Public API (Swagger UI)
- **Live Swagger UI:** [https://linear-regression-model-ry4e.onrender.com/docs](https://linear-regression-model-ry4e.onrender.com/docs)

*(Note: This service runs on Render's free tier, so it may take 30–60 seconds to wake up on the first request after 15 minutes of inactivity.)*

## Video Demo
- **YouTube Demo:** [[YouTube video link](https://youtu.be/W3usJYcraVQ)](https://youtu.be/5YCl2j8Olt4?si=tkj1rT2PBHRfPQ_K)

## How to Run the Mobile App

### Prerequisites
- [Flutter SDK](https://docs.flutter.dev/get-started/install) installed and added to your PATH
- An Android or iOS emulator running (or a physical phone connected with USB debugging enabled)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/bendoujanna/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
    ```

2. Install Flutter dependencies
    ```
    flutter pub get
    ```

3. Check that your emulator or phone is detected
    ```
    flutter devices
    ```

4. Launch app
    ```
    flutter run
    ```

***Fill in all 9 fields (enter a whole number for Length of stay and pick options for the remaining 8 fields), then tap Predict to get the estimated cost from the live Render API**

