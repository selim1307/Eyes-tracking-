import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Étape 1 : Charger les données ===
csv_file = "features_log.csv"  # <-- Change le nom si besoin
df = pd.read_csv(csv_file)

# === Étape 2 : Préparer les features ===
columns_to_drop = [col for col in ["timestamp", "gaze", "anomaly"] if col in df.columns]
features = df.drop(columns=columns_to_drop)

# === Étape 3 : Charger le modèle ===
model = joblib.load("eye_anomaly_model.pkl")

# === Étape 4 : Faire les prédictions ===
predictions = model.predict(features)

# Ajouter les résultats au dataframe
df["anomaly"] = predictions
df["anomaly_label"] = df["anomaly"].map({-1: "🚨 Anomalie", 1: "✅ Normal"})

# === Étape 5 : Sauvegarder les résultats ===
df.to_csv("anomaly_log.csv", index=False)
print("✅ Anomalies détectées et sauvegardées dans 'anomaly_log.csv'.")

# === Étape 6 : Afficher un résumé ===
anomalies = df[df["anomaly"] == -1]
print(f"\nNombre d’anomalies détectées : {len(anomalies)}")
if "timestamp" in df.columns:
    print(anomalies[["timestamp", "anomaly_label"]])
else:
    print(anomalies[["anomaly_label"]])

# === Visualisation des anomalies ===
if "timestamp" in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_anomaly'] = df['anomaly'] == -1

    # Obtenir les scores d’anomalie
    scores = model.decision_function(features)  # Plus bas = plus anormal
    df["anomaly_score"] = -scores  # On inverse pour que + élevé = plus anormal

    # Nouveau graphe : score d'anomalie dans le temps
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['anomaly_score'], label="Score d'anomalie", color='blue')
    plt.scatter(df[df.is_anomaly]['timestamp'], df[df.is_anomaly]['anomaly_score'], color='red', label="🚨 Anomalie")
    plt.title("Score d’anomalie au fil du temps")
    plt.xlabel("Temps")
    plt.ylabel("Score d’anomalie (plus élevé = plus suspect)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Charger les données ===
df = pd.read_csv("features_log.csv")

# === Nettoyer les colonnes ===
features = df.drop(columns=[col for col in ["timestamp", "gaze", "anomaly"] if col in df.columns])

# === Étiquettes (s’il y a 'anomaly' dans le CSV) ===
labels = df["anomaly"] if "anomaly" in df.columns else None

# === Standardiser les données ===
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# === Appliquer PCA (2 composants pour visualisation) ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# === Création d’un DataFrame PCA ===
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
if labels is not None:
    pca_df["anomaly"] = labels

# === Visualisation ===
plt.figure(figsize=(10, 6))
if labels is not None:
    plt.scatter(pca_df[pca_df["anomaly"] == 1]["PC1"],
                pca_df[pca_df["anomaly"] == 1]["PC2"],
                label="✅ Normal", alpha=0.6)
    plt.scatter(pca_df[pca_df["anomaly"] == -1]["PC1"],
                pca_df[pca_df["anomaly"] == -1]["PC2"],
                label="🚨 Anomalie", alpha=0.6, color="red")
else:
    plt.scatter(pca_df["PC1"], pca_df["PC2"], label="Données")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA des features de mouvement oculaire")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
