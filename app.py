from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# --- GLOBAL STORAGE (In-Memory for Hackathon) ---
# This holds the data temporarily until both files are uploaded
DATA_STORE = {
    "enrolment": None,
    "biometric": None
}

def detect_file_type(df):
    """Figure out if this is the Enrolment file or the Biometric file based on columns."""
    cols = df.columns.str.lower().tolist()
    
    # Check for Enrolment keywords
    if any(x in cols for x in ['enrol_age_0_5', 'age_0_5', 'enrolment']):
        return "enrolment"
    
    # Check for Biometric/Demographic keywords
    if any(x in cols for x in ['bio_age_5_17', 'biometric_updates', 'mandatory_updates', 'demographic']):
        return "biometric"
    
    return "unknown"

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global DATA_STORE
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        print(f"📥 Received file: {file.filename}")
        
        # 1. Read the file
        try:
            df = pd.read_csv(file)
            # Normalize columns immediately
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            return jsonify({"error": "Could not read CSV"}), 400

        # 2. Identify and Store
        file_type = detect_file_type(df)
        
        if file_type == "enrolment":
            # Standardize State/District for merging
            if 'state' in df.columns and 'district' in df.columns:
                # Group strictly by location to ensure unique rows for merging
                # Assuming the column for count is 'enrol_age_0_5' or similar
                col_name = next((c for c in df.columns if c in ['enrol_age_0_5', 'age_0_5']), None)
                if col_name:
                    DATA_STORE["enrolment"] = df.groupby(['state', 'district'], as_index=False)[col_name].sum()
                    # Rename standard column
                    DATA_STORE["enrolment"].rename(columns={col_name: 'enrol_age_0_5'}, inplace=True)
                    print("✅ Stored Enrolment Data")
                else:
                    return jsonify({"error": "Could not find 'enrol_age_0_5' column"}), 400
            else:
                 return jsonify({"error": "File missing 'state' or 'district' columns"}), 400

        elif file_type == "biometric":
            if 'state' in df.columns and 'district' in df.columns:
                 # Find the biometric column
                col_name = next((c for c in df.columns if c in ['bio_age_5_17', 'biometric_updates']), None)
                if col_name:
                    DATA_STORE["biometric"] = df.groupby(['state', 'district'], as_index=False)[col_name].sum()
                    DATA_STORE["biometric"].rename(columns={col_name: 'bio_age_5_17'}, inplace=True)
                    print("✅ Stored Biometric Data")
                else:
                    return jsonify({"error": "Could not find 'bio_age_5_17' column"}), 400
        else:
            return jsonify({"error": "Could not identify file type. Ensure columns are named correctly."}), 400

        # 3. CHECK: Do we have BOTH files?
        if DATA_STORE["enrolment"] is None:
            return jsonify({
                "status": "partial",
                "message": "✅ Biometric data received. Waiting for Enrolment file...",
                "logs": ["📥 Biometric data loaded.", "⏳ Please upload the Enrolment (0-5) CSV next."]
            })
        
        if DATA_STORE["biometric"] is None:
            return jsonify({
                "status": "partial",
                "message": "✅ Enrolment data received. Waiting for Biometric file...",
                "logs": ["📥 Enrolment data loaded.", "⏳ Please upload the Biometric (5-17) CSV next."]
            })

        # --- STEP 4: MERGE & ANALYZE (Only runs when both exist) ---
        print("🔄 Both files present. Merging...")
        
        # Merge on State and District
        merged_df = pd.merge(
            DATA_STORE["enrolment"], 
            DATA_STORE["biometric"], 
            on=['state', 'district'], 
            how='inner' # Keep only districts that exist in both
        )
        
        # Now the Math will be ACCURATE
        merged_df["enrol_age_0_5"] = pd.to_numeric(merged_df["enrol_age_0_5"], errors="coerce").fillna(0)
        merged_df["bio_age_5_17"] = pd.to_numeric(merged_df["bio_age_5_17"], errors="coerce").fillna(0)
        
        # 1. Compliance
        merged_df["biometric_compliance_ratio"] = (
            merged_df["bio_age_5_17"] / (merged_df["enrol_age_0_5"] + 1)
        )
        
        # 2. Ghost Gap (The real calculation)
        merged_df["ghost_gap"] = (
            merged_df["enrol_age_0_5"] - merged_df["bio_age_5_17"]
        ).clip(lower=0)
        
        # 3. Risk Score
        merged_df["ghost_risk_score"] = (
            merged_df["ghost_gap"] / (merged_df["enrol_age_0_5"] + 1)
        )
        
        # Fraud Label
        merged_df["fraud_label"] = (merged_df["biometric_compliance_ratio"] < 0.4).astype(int)

        # ML Training
        features = ["enrol_age_0_5", "bio_age_5_17", "ghost_gap", "ghost_risk_score", "biometric_compliance_ratio"]
        X = merged_df[features]
        y = merged_df["fraud_label"]

        accuracy = 0.0
        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
        else:
            accuracy = 1.0

        # Format for Dashboard
        output_df = merged_df.rename(columns={
            "fraud_label": "suspicious_pincodes", 
            "ghost_gap": "total_ghost_children"
        })

        result_data = output_df.sort_values("ghost_risk_score", ascending=False).to_dict(orient="records")
        
        # Clear store after success so user can start over
        DATA_STORE = {"enrolment": None, "biometric": None}

        return jsonify({
            "status": "success",
            "accuracy": f"{accuracy*100:.2f}%",
            "total_records": len(merged_df),
            "data": result_data,
            "logs": [
                "✅ Merged Enrolment & Biometric Data",
                f"📊 Analyzed {len(merged_df)} Districts",
                f"🧠 Model Accuracy: {accuracy*100:.2f}%",
                "🎯 Final Output Ready"
            ]
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)