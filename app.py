from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import io

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# ==========================
# LOAD MODEL & SCALER
# ==========================
# Use absolute paths for Render
import os

# Get the directory where app.py is located
basedir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(basedir, "random_forest_diabetes_model.pkl")
scaler_path = os.path.join(basedir, "scaler_5000.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

latest_report = {}

# ==========================
# HEALTH CHECK FUNCTION
# ==========================
def get_health_status(patient_raw):
    report = []
    checks = {
        "BMI": (18.5, 24.9),
        "Glucose": (70, 99),
        "BloodPressure": (90, 130),
        "ExerciseHours": (3, 7),
        "Cholesterol": (125, 200),
        "SleepHours": (7, 9),
        "HbA1c": (4.0, 5.7)
    }
    
    for feature, (low, high) in checks.items():
        if feature in patient_raw.columns:
            value = float(patient_raw[feature].iloc[0])
            if value < low:
                status = "Low"
                icon = "🔻"
            elif value > high:
                status = "High"
                icon = "🔺"
            else:
                status = "Normal"
                icon = "✅"
            
            report.append({
                "Feature": feature,
                "Value": round(value, 1),
                "Range": f"{low} - {high}",
                "Status": f"{icon} {status}",
                "StatusText": status
            })
    return report

# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global latest_report
    
    try:
        patient_name = request.form["PatientName"].strip()
        
        raw_values = {
            "Age": float(request.form["Age"]),
            "BMI": float(request.form["BMI"]),
            "Glucose": float(request.form["Glucose"]),
            "BloodPressure": float(request.form["BloodPressure"]),
            "ExerciseHours": float(request.form["ExerciseHours"]),
            "FamilyHistory": int(request.form["FamilyHistory"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "SleepHours": float(request.form["SleepHours"]),
            "Smoking": int(request.form["Smoking"]),
            "HbA1c": float(request.form["HbA1c"])
        }
        
        patient_raw = pd.DataFrame([raw_values])
        
        # Create SCALED version for model
        patient_scaled = patient_raw.copy()
        numerical_cols = ["Age", "BMI", "Glucose", "BloodPressure", "ExerciseHours", 
                         "Cholesterol", "SleepHours", "HbA1c"]
        
        patient_scaled[numerical_cols] = scaler.transform(patient_raw[numerical_cols])
        
        # Make prediction
        probability = model.predict_proba(patient_scaled)[0][1]
        
        if probability < 0.30:
            risk = "Low Risk"
            risk_class = "low"
        elif probability < 0.60:
            risk = "Moderate Risk"
            risk_class = "moderate"
        else:
            risk = "High Risk"
            risk_class = "high"
        
        health_report = get_health_status(patient_raw)
        
        latest_report = {
            "patient_name": patient_name,
            "raw_values": raw_values,
            "probability": round(probability * 100, 2),
            "risk": risk,
            "risk_class": risk_class,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "health_report": health_report
        }
        
        # Save to history
        history_path = os.path.join(basedir, "patient_history.csv")
        history_entry = pd.DataFrame([{
            "PatientName": patient_name,
            "Age": raw_values["Age"],
            "BMI": raw_values["BMI"],
            "Glucose": raw_values["Glucose"],
            "BloodPressure": raw_values["BloodPressure"],
            "ExerciseHours": raw_values["ExerciseHours"],
            "FamilyHistory": raw_values["FamilyHistory"],
            "Cholesterol": raw_values["Cholesterol"],
            "SleepHours": raw_values["SleepHours"],
            "Smoking": raw_values["Smoking"],
            "HbA1c": raw_values["HbA1c"],
            "Probability_%": round(probability * 100, 2),
            "Risk": risk,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        if os.path.exists(history_path):
            existing = pd.read_csv(history_path)
            updated = pd.concat([existing, history_entry], ignore_index=True)
            updated.to_csv(history_path, index=False)
        else:
            history_entry.to_csv(history_path, index=False)
        
        return render_template(
            "result.html",
            patient_name=patient_name,
            probability=round(probability * 100, 2),
            risk=risk,
            risk_class=risk_class,
            health_report=health_report,
            raw_values=raw_values
        )
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/history")
def history():
    history_path = os.path.join(basedir, "patient_history.csv")
    if not os.path.exists(history_path):
        return render_template("history.html", patients=[], message="No records found")
    
    df = pd.read_csv(history_path)
    patients = df.to_dict('records')
    return render_template("history.html", patients=patients, message=None)

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        search_name = request.form.get("PatientName", "").strip().lower()
        history_path = os.path.join(basedir, "patient_history.csv")
        
        if not os.path.exists(history_path):
            return render_template("search.html", results=[], search_term=search_name, message="No records found")
        
        df = pd.read_csv(history_path)
        results = df[df["PatientName"].str.lower().str.contains(search_name, na=False)]
        
        if len(results) > 0:
            patients = results.to_dict('records')
            return render_template("search.html", results=patients, search_term=search_name, message=None)
        else:
            return render_template("search.html", results=[], search_term=search_name, message=f"No patients found matching '{search_name}'")
    
    return render_template("search.html", results=None, search_term=None, message=None)

@app.route("/download_report")
def download_report():
    if not latest_report:
        return "No report available. Please make a prediction first."
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []
    
    content.append(Paragraph("Diabetes Risk Assessment Report", styles["Title"]))
    content.append(Spacer(1, 12))
    
    content.append(Paragraph(f"Patient Name: {latest_report['patient_name']}", styles["Normal"]))
    content.append(Paragraph(f"Report Date: {latest_report['timestamp']}", styles["Normal"]))
    content.append(Paragraph(f"Diabetes Probability: {latest_report['probability']}%", styles["Normal"]))
    content.append(Paragraph(f"Risk Category: {latest_report['risk']}", styles["Normal"]))
    content.append(Spacer(1, 12))
    
    # Health table
    table_data = [["Feature", "Your Value", "Normal Range", "Status"]]
    for row in latest_report["health_report"]:
        table_data.append([row["Feature"], str(row["Value"]), row["Range"], row["Status"]])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    
    content.append(table)
    doc.build(content)
    
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"{latest_report['patient_name']}_report.pdf", mimetype='application/pdf')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)