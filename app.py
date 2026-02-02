import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Page settings
# ----------------------------
st.set_page_config(page_title="Shipment Delay Predictor", layout="centered")
st.title("🚚 Shipment Delay Predictor + Decision Assistant")
st.write("Enter shipment details → click **Predict Delay Risk** → get **delay probability** + **recommended actions**.")

# ----------------------------
# Load trained model (robust path)
# ----------------------------
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

# ----------------------------
# Helper functions
# ----------------------------
def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

def recommend_actions(r: dict, prob: float):
    actions = []

    # Probability-driven actions
    if prob >= 0.70:
        actions.append("High risk: Alert operations team immediately and prioritize this shipment.")
    elif prob >= 0.40:
        actions.append("Medium risk: Add buffer time and monitor status closely.")
    else:
        actions.append("Low risk: Proceed as planned with standard monitoring.")

    # Feature-driven actions
    if r["weather"] in ["Storm", "Snow"]:
        actions.append("Add additional buffer time due to severe weather risk.")
    if r["transport_mode"] == "Train":
        actions.append("Consider switching to Truck for flexibility (if SLA is strict).")
    if r["distance_km"] > 1200:
        actions.append("Consider hub transfer / splitting route to reduce long-haul delay risk.")
    if r["planned_days"] < 2 and r["distance_km"] > 900:
        actions.append("Planned time seems aggressive for distance—adjust ETA to reduce SLA breach.")

    # Remove duplicates while preserving order
    seen = set()
    final_actions = []
    for a in actions:
        if a not in seen:
            final_actions.append(a)
            seen.add(a)

    return final_actions

# ----------------------------
# Inputs
# ----------------------------
st.subheader("🧾 Shipment Details")

origin = st.selectbox(
    "Origin",
    ["Dallas", "Atlanta", "Los Angeles", "Seattle", "Houston", "Miami", "Chicago", "San Jose", "Phoenix", "Boston", "Denver"]
)

destination = st.selectbox(
    "Destination",
    ["Chicago", "New York", "Phoenix", "San Francisco", "Denver", "Orlando", "Detroit", "San Diego",
     "Las Vegas", "Philadelphia", "Austin", "St Louis", "Miami", "San Jose"]
)

carrier = st.selectbox("Carrier", ["FedEx", "UPS", "DHL"])
transport_mode = st.selectbox("Transport Mode", ["Truck", "Train"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Storm", "Snow"])

distance_km = st.number_input("Distance (km)", min_value=0, value=800, step=10)
planned_days = st.number_input("Planned Days", min_value=1, value=3, step=1)
actual_days = st.number_input("Actual Days (if unknown, set = planned)", min_value=1, value=3, step=1)

# Build input row
row = pd.DataFrame([{
    "origin": origin,
    "destination": destination,
    "distance_km": distance_km,
    "carrier": carrier,
    "transport_mode": transport_mode,
    "weather": weather,
    "planned_days": planned_days,
    "actual_days": actual_days
}])

# ----------------------------
# Predict button
# ----------------------------
st.divider()

if st.button("Predict Delay Risk"):
    prob = float(model.predict_proba(row)[:, 1][0])
    label = risk_label(prob)

    st.subheader("📌 Prediction")

    # Show probability + label
    st.metric("Delay Risk Probability", f"{prob:.2f}", help="Probability that the shipment will be delayed (class=1).")
    st.write(f"**Risk Level:** `{label}`")

    # Visual bar (simple)
    st.progress(min(max(prob, 0.0), 1.0))

    # Recommended actions
    st.subheader("✅ Recommended Actions")
    for a in recommend_actions(row.iloc[0].to_dict(), prob):
        st.write("•", a)

    # Show input row for transparency
    with st.expander("Show input data"):
        st.dataframe(row)

else:
    st.info("Fill the shipment details above, then click **Predict Delay Risk**.")

