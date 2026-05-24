import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def generate_simulated_data(num_records=1000):
    """Genera un dataset simulado de comportamiento de clientes."""
    np.random.seed(42)
    data = {
        'Tenure_Months': np.random.randint(1, 72, num_records),
        'Monthly_Charges': np.random.uniform(20.0, 120.0, num_records),
        'Total_Transactions': np.random.randint(10, 500, num_records),
        'Support_Tickets': np.random.randint(0, 10, num_records)
    }
    df = pd.DataFrame(data)

    # Lógica de simulación: Más tickets y cargos altos = mayor probabilidad de churn
    churn_prob = (df['Support_Tickets'] * 0.1) + (df['Monthly_Charges'] / 200) - (df['Tenure_Months'] / 100)
    df['Churn'] = np.where(churn_prob > np.random.uniform(0.3, 0.8, num_records), 1, 0)
    return df

def run_analysis():
    print("Iniciando pipeline de análisis de datos...")
    df = generate_simulated_data()

    # 1. Preparación de Datos
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Entrenamiento del Modelo Predictivo
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 3. Evaluación
    predictions = model.predict(X_test)
    print("\n--- Resultados del Modelo de Retención ---")
    print(classification_report(y_test, predictions))

    # 4. Extracción de Insights
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Factores Clave de Abandono (Feature Importance) ---")
    print(feature_importance)

if __name__ == "__main__":
    run_analysis()
