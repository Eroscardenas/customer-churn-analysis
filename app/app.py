from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------------------------------------------------
# Load data and trained model
# ---------------------------------------------------------

df = pd.read_csv("data/processed/telco_customer_churn_clean.csv")

model = joblib.load("models/best_churn_model.joblib")

MODEL_FEATURES = list(model.feature_names_in_)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def yes_no_features(prefix, value):
    """
    Generates one-hot encoded Yes/No columns.
    Example:
    tech_support = Yes
    ->
    tech_support_yes = 1
    tech_support_no = 0
    """
    return {
        f"{prefix}_no": 1 if value == "No" else 0,
        f"{prefix}_yes": 1 if value == "Yes" else 0,
    }


def build_model_input(form):
    tenure = int(form["tenure"])
    monthly_charges = float(form["monthly_charges"])
    total_charges = float(form["total_charges"])

    gender = form["gender"]
    senior_citizen = form["senior_citizen"]
    partner = form["partner"]
    dependents = form["dependents"]

    phone_service = form["phone_service"]
    multiple_lines = form["multiple_lines"]

    internet_service = form["internet_service"]

    online_security = form["online_security"]
    online_backup = form["online_backup"]
    device_protection = form["device_protection"]
    tech_support = form["tech_support"]
    streaming_tv = form["streaming_tv"]
    streaming_movies = form["streaming_movies"]

    contract = form["contract"]
    paperless_billing = form["paperless_billing"]
    payment_method = form["payment_method"]

    # Start all model features at zero
    data = {feature: 0 for feature in MODEL_FEATURES}

    # -----------------------------------------------------
    # Numerical features
    # -----------------------------------------------------

    data["tenure"] = tenure
    data["monthly_charges"] = monthly_charges
    data["total_charges"] = total_charges

    # -----------------------------------------------------
    # Derived features
    # -----------------------------------------------------

    data["long_term_contract"] = 1 if contract in ["One year", "Two year"] else 0

    data["automatic_payment"] = 1 if payment_method in [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] else 0

    data["has_internet"] = 0 if internet_service == "No" else 1

    # Count active subscribed services
    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]

    data["number_of_services"] = sum(
        1 for value in service_values if value == "Yes"
    )

    # -----------------------------------------------------
    # Gender
    # -----------------------------------------------------

    data["gender_female"] = 1 if gender == "Female" else 0
    data["gender_male"] = 1 if gender == "Male" else 0

    # -----------------------------------------------------
    # Senior citizen
    # -----------------------------------------------------

    data["senior_citizen_no"] = 1 if senior_citizen == "No" else 0
    data["senior_citizen_yes"] = 1 if senior_citizen == "Yes" else 0

    # -----------------------------------------------------
    # Partner
    # -----------------------------------------------------

    data["partner_no"] = 1 if partner == "No" else 0
    data["partner_yes"] = 1 if partner == "Yes" else 0

    # -----------------------------------------------------
    # Dependents
    # -----------------------------------------------------

    data["dependents_no"] = 1 if dependents == "No" else 0
    data["dependents_yes"] = 1 if dependents == "Yes" else 0

    # -----------------------------------------------------
    # Phone service
    # -----------------------------------------------------

    data["phone_service_no"] = 1 if phone_service == "No" else 0
    data["phone_service_yes"] = 1 if phone_service == "Yes" else 0

    # -----------------------------------------------------
    # Multiple lines
    # -----------------------------------------------------

    if phone_service == "No":
        data["multiple_lines_no_phone_service"] = 1
    elif multiple_lines == "Yes":
        data["multiple_lines_yes"] = 1
    else:
        data["multiple_lines_no"] = 1

    # -----------------------------------------------------
    # Internet service
    # -----------------------------------------------------

    if internet_service == "DSL":
        data["internet_service_dsl"] = 1
    elif internet_service == "Fiber optic":
        data["internet_service_fiber_optic"] = 1
    else:
        data["internet_service_no"] = 1

    # -----------------------------------------------------
    # Internet-dependent services
    # -----------------------------------------------------

    internet_services = {
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
    }

    for prefix, value in internet_services.items():

        if internet_service == "No":
            data[f"{prefix}_no_internet_service"] = 1

        elif value == "Yes":
            data[f"{prefix}_yes"] = 1

        else:
            data[f"{prefix}_no"] = 1

    # -----------------------------------------------------
    # Contract
    # -----------------------------------------------------

    if contract == "Month-to-month":
        data["contract_month_to_month"] = 1

    elif contract == "One year":
        data["contract_one_year"] = 1

    elif contract == "Two year":
        data["contract_two_year"] = 1

    # -----------------------------------------------------
    # Paperless billing
    # -----------------------------------------------------

    data["paperless_billing_no"] = 1 if paperless_billing == "No" else 0
    data["paperless_billing_yes"] = 1 if paperless_billing == "Yes" else 0

    # -----------------------------------------------------
    # Payment method
    # -----------------------------------------------------

    payment_mapping = {
        "Bank transfer (automatic)":
            "payment_method_bank_transfer_automatic",

        "Credit card (automatic)":
            "payment_method_credit_card_automatic",

        "Electronic check":
            "payment_method_electronic_check",

        "Mailed check":
            "payment_method_mailed_check",
    }

    data[payment_mapping[payment_method]] = 1

    # -----------------------------------------------------
    # Tenure groups
    # -----------------------------------------------------

    if tenure <= 12:
        data["tenure_group_0_12_months"] = 1

    elif tenure <= 24:
        data["tenure_group_13_24_months"] = 1

    elif tenure <= 48:
        data["tenure_group_25_48_months"] = 1

    else:
        data["tenure_group_49_72_months"] = 1

    # Build dataframe in exact order expected by model
    model_input = pd.DataFrame(
        [[data[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )

    return model_input


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

@app.route("/")
def index():

    total_customers = len(df)

    churned_customers = len(
        df[df["churn"] == "Yes"]
    )

    retained_customers = len(
        df[df["churn"] == "No"]
    )

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    avg_monthly_charges = df["monthly_charges"].mean()

    # Churn rate by contract
    contract_churn = (
        df.assign(
            churn_flag=(df["churn"] == "Yes").astype(int)
        )
        .groupby("contract")["churn_flag"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )

    # Churn rate by internet service
    internet_churn = (
        df.assign(
            churn_flag=(df["churn"] == "Yes").astype(int)
        )
        .groupby("internet_service")["churn_flag"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )

    return render_template(
        "index.html",

        total_customers=total_customers,
        churned_customers=churned_customers,
        retained_customers=retained_customers,
        churn_rate=round(churn_rate, 1),
        avg_monthly_charges=round(avg_monthly_charges, 2),

        contract_labels=contract_churn.index.tolist(),
        contract_values=contract_churn.values.tolist(),

        internet_labels=internet_churn.index.tolist(),
        internet_values=internet_churn.values.tolist(),
    )


# ---------------------------------------------------------
# Churn predictor
# ---------------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    probability = None
    risk_level = None

    if request.method == "POST":

        try:

            model_input = build_model_input(request.form)

            prediction_value = int(
                model.predict(model_input)[0]
            )

            probability = float(
                model.predict_proba(model_input)[0][1]
            )

            probability = round(
                probability * 100,
                1
            )

            prediction = (
                "Likely to Churn"
                if prediction_value == 1
                else "Likely to Stay"
            )

            if probability >= 70:
                risk_level = "High"

            elif probability >= 40:
                risk_level = "Medium"

            else:
                risk_level = "Low"

        except Exception as error:

            prediction = "Prediction Error"
            probability = None
            risk_level = None

            print("Prediction error:", error)

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
    )


# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)