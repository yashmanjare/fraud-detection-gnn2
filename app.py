import streamlit as st
import pandas as pd
import numpy as np
# Import the correct function name
# from prediction_function import predict_fraud

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}
.main {
    padding: 1.5rem 3rem;
}
h1, h2, h3 {
    color: #1f2937;
}
.stButton>button {
    background: linear-gradient(to right, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #1e40af, #1d4ed8);
    transform: scale(1.02);
}
.dataframe th {
    background-color: #2563eb !important;
    color: white !important;
}
.metric-card {
    background-color: white;
    border-radius: 15px;
    padding: 1.2rem;
    box_shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #2563eb;
}
.metric-label {
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("💳 Intelligent Fraud Detection System")
st.write("Upload transaction data to analyze and detect **risky or fraudulent transactions** in real-time using AI.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])

# --- Default model path ---
model_path = "gcn_correlation_smote_model.pth"

# --- Process the Uploaded CSV ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head(8))
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    # --- Run Prediction ---
    if st.button("🚀 Run Fraud Detection", use_container_width=True):
        with st.spinner("Analyzing transactions..."):
            try:
                # Call the correct prediction function and handle its return value
                # predict_fraud returns a DataFrame with original data + predictions
                out_df = predict_fraud(
                    df,
                    model_path=model_path,
                )

                if out_df is None:
                     st.error("⚠️ Prediction failed. Please check the console for details.")
                     st.stop()

            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred during prediction: {e}")
                st.stop()

        # Assuming the prediction was successful and returned a DataFrame
        st.success("✅ Detection completed.")

        # --- Summary Metrics ---
        st.subheader("📈 Detection Summary")

        # Use the 'Predicted_Class' column added by predict_fraud
        fraud_count = (out_df['Predicted_Class'] == 1).sum()
        total_count = len(out_df)
        fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_count}</div><div class='metric-label'>Total Transactions</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{fraud_count}</div><div class='metric-label'>Fraudulent Transactions</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{fraud_rate:.2f}%</div><div class='metric-label'>Fraud Rate</div></div>", unsafe_allow_html=True)

        # --- Risky Transactions ---
        # Filter based on 'Predicted_Class' == 1 and sort by 'Fraud_Probability'
        risky = out_df[out_df['Predicted_Class'] == 1].sort_values('Fraud_Probability', ascending=False)
        st.subheader("🚨 Risky Transactions Detected")
        if not risky.empty:
            # Display relevant columns from the risky transactions
            st.dataframe(risky[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Predicted_Class', 'Fraud_Probability']].head(50))
        else:
            st.success("🎉 No risky transactions detected.")

        # --- Download Results ---
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Full Prediction Report", csv, "fraud_predictions.csv", "text/csv")

else:
    st.info("⬆️ Upload a CSV file to begin fraud analysis.")
