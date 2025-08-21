from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import pickle as pk
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Define and create necessary folders
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PREDICTIONS_FOLDER = os.path.join(app.instance_path, 'predictions')
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER  # Fix for KeyError

# Load trained model and scaler
try:
    model = pk.load(open("model.plk", "rb"))  
    scaler = pk.load(open("scaler.plk", "rb"))
except FileNotFoundError:
    print("Error: Model or scaler file not found. Ensure 'model.plk' and 'scaler.plk' exist.")
    model, scaler = None, None

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# Single Loan Prediction Page
@app.route("/single", methods=["GET"])
def single():
    return render_template("single.html", form_data={})

# Single Loan Prediction Processing
@app.route("/predict_single", methods=["GET"])
def predict_single():
    form_data = request.args

    if "clear" in form_data:
        return render_template("single.html", result=None, suggestion=None, form_data={})

    if not form_data:
        return render_template("single.html", result=None, form_data={})

    try:
        # Extract input values
        no_of_dep = int(form_data.get("no_of_dep", 0))
        education = form_data.get("education", "Graduated")
        self_employed = form_data.get("self_employed", "No")
        income_annum = float(form_data.get("income_annum", 0))
        loan_amount = float(form_data.get("loan_amount", 0))
        loan_term = int(form_data.get("loan_term", 0))
        cibil_score = int(form_data.get("cibil_score", 0))
        assets = float(form_data.get("assets", 0))

        # Encode categorical values
        education_encoded = 0 if education == "Graduated" else 1
        self_employed_encoded = 0 if self_employed == "No" else 1

        # Prepare input data
        pred_data = pd.DataFrame([[no_of_dep, education_encoded, self_employed_encoded, income_annum, 
                                   loan_amount, loan_term, cibil_score, assets]],
                                 columns=["no_of_dependents", "education", "self_employed", "income_annum",
                                          "loan_amount", "loan_term", "cibil_score", "Assets"])
        
        # Scale input data
        pred_data = scaler.transform(pred_data)

        # Make Prediction
        prediction = model.predict(pred_data)

        # Loan Suggestion if rejected
        def suggest_loan(income_annum, cibil_score, loan_amount, assets):
            if cibil_score >= 750:
                max_loan = income_annum * 1.5
            elif cibil_score >= 650:
                max_loan = income_annum * 1.3
            elif cibil_score >= 550:
                max_loan = income_annum * 1.2
            elif cibil_score >= 470:
                max_loan = income_annum * 0.6
            else:
                return "Your CIBIL score is too low for loan approval."

            if max_loan >= assets:
                return "Your asset value is too low for the requested loan amount."
            elif max_loan < loan_amount:
                return f"Loan is rejected. You may qualify for a loan up to: {max_loan:.2f}"
            else:
                return "Try increasing cibil score, asset, and loan duration for better approval chances."

        # Generate Result
        if prediction[0] == 1:
            result = "✅ Loan is Approved!"
            suggestion = None
        else:
            result = "❌ Loan is Rejected!"
            suggestion = suggest_loan(income_annum, cibil_score, loan_amount, assets)

    except Exception as e:
        result = f"Error: {str(e)}"
        suggestion = None

    return render_template("single.html", result=result, suggestion=suggestion, form_data=form_data)

# File-based Loan Prediction
@app.route("/index", methods=["GET"])
def index():
    return render_template("file.html")

# Process uploaded file
def process_file(file_path, file_ext, scale=False, for_preview=False):
    if file_ext == "csv":
        df = pd.read_csv(file_path)
    elif file_ext in ["xls", "xlsx"]:
        df = pd.read_excel(file_path)
    else:
        return None

    required_columns = ["name", "no_of_dependents", "education", "self_employed", "income_annum",
                        "loan_amount", "loan_term", "cibil_score", "Assets"]

    if not all(col in df.columns for col in required_columns):
        return None  
     
    df_original = df.copy()

    if for_preview:
        return df_original  

    # Convert categorical data
    df["education"] = df["education"].map({"Graduated": 0, "Not Graduated": 1}).fillna(0)
    df["self_employed"] = df["self_employed"].map({"No": 0, "Yes": 1}).fillna(0)
    
    # Fill missing numerical values
    df = df.fillna(0)

    numerical_cols = ["no_of_dependents", "education", "self_employed", "income_annum",
                      "loan_amount", "loan_term", "cibil_score", "Assets"]

    df_numeric = df[numerical_cols].copy()

    if scale:
        df_numeric = df_numeric.astype(float)  
        df_numeric[numerical_cols] = scaler.transform(df_numeric[numerical_cols])

    df_numeric.insert(0, "name", df["name"])
    
    return df_numeric, df_original

# Predict loan status
def predict_loan_status(df_scaled, df_original):
    try:
        predictions = model.predict(df_scaled.drop(columns=["name"]))
        df_original["loan_status"] = ["Approved" if p == 1 else "Rejected" for p in predictions]
    except Exception as e:
        print("Prediction error:", e)
        return None

    return df_original

# File Preview
@app.route('/preview', methods=['POST'])
def preview():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ["csv", "xls", "xlsx"]:
        return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    original_df = process_file(file_path, file_ext, for_preview=True)
    if original_df is None:
        return jsonify({"error": "Failed to process file. Ensure correct format and columns."})

    return jsonify({"preview": original_df.to_dict(orient='records')})

# Predict loan status
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ["csv", "xls", "xlsx"]:
        return jsonify({"error": "Unsupported file format."})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    df_scaled, df_original = process_file(file_path, file_ext, scale=True)

    if df_scaled is None or df_original is None:
        return jsonify({"error": "Failed to process file."})

    df_with_predictions = predict_loan_status(df_scaled, df_original)

    prediction_file = os.path.join(app.config['PREDICTIONS_FOLDER'], "predictions.csv")
    df_with_predictions.to_csv(prediction_file, index=False)

    return jsonify({"predictions": df_with_predictions.to_dict(orient='records')})

# Download predictions
@app.route('/download', methods=['GET'])
def download():
    return send_file(os.path.join(app.config['PREDICTIONS_FOLDER'], "predictions.csv"), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
