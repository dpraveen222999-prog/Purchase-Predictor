from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

app = Flask(__name__)

# ---- Train Model ----
df = pd.read_csv(r"C:\Users\dprav\Downloads\shopping_trends.csv")

# Encode target
le = LabelEncoder()
df['Subscription Status'] = le.fit_transform(df['Subscription Status'])

# One-hot encode
df = pd.get_dummies(df, columns=['Discount Applied', 'Promo Code Used'])

features = ['Age','Purchase Amount (USD)','Review Rating','Previous Purchases',
            'Discount Applied_Yes','Discount Applied_No',
            'Promo Code Used_Yes','Promo Code Used_No']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X_train, X_test, y_train, y_test = train_test_split(df[features], df['Subscription Status'], test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

best_model, best_auc, best_name = None, -1, None
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    if auc > best_auc:
        best_model, best_auc, best_name = model, auc, name


# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None
    if request.method == "POST":
        age = int(request.form["age"])
        amt = float(request.form["amt"])
        rate = float(request.form["rate"])
        prev = int(request.form["prev"])
        disc = request.form["disc"]
        promo = request.form["promo"]

        new_data = pd.DataFrame([{
            'Age': age,
            'Purchase Amount (USD)': amt,
            'Review Rating': rate,
            'Previous Purchases': prev,
            'Discount Applied_Yes': 1 if disc=="Yes" else 0,
            'Discount Applied_No': 1 if disc=="No" else 0,
            'Promo Code Used_Yes': 1 if promo=="Yes" else 0,
            'Promo Code Used_No': 1 if promo=="No" else 0
        }])

        new_data[features] = scaler.transform(new_data[features])
        pred = best_model.predict(new_data)[0]
        prob = best_model.predict_proba(new_data)[:,1][0]

        prediction = "WILL BUY ✅" if pred==1 else "WILL NOT BUY ❌"
        confidence = f"{prob:.2f}"

    return render_template("index.html", model_name=best_name, auc=f"{best_auc:.3f}", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
