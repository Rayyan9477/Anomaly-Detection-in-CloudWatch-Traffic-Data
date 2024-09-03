import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Load the dataset

df = pd.read_csv('CloudWatch_Traffic_Web_Attack.csv')

df.head(10)


# Data preprocessing
print("Preprocessing the data...")
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
print("Data preprocessing completed.")


# Extract relevant features
print("Extracting relevant features...")
features = ['bytes_in', 'bytes_out', 'duration']
X = df[features].values


# Normalize the features
print("Normalizing the features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature normalization completed.")

# Perform anomaly detection using Isolation Forest
print("Performing anomaly detection using Isolation Forest...")
clf = IsolationForest(contamination='auto')
y_pred = clf.fit_predict(X_scaled)
print("Anomaly detection completed.")


# Identify anomalies
print("Identifying anomalies...")
anomalies = df.loc[y_pred == -1]
print("Anomaly identification completed.")

# Print the anomalies
print("Detected anomalies:")
print(anomalies[['creation_time', 'end_time', 'src_ip', 'dst_ip', 'bytes_in', 'bytes_out', 'duration']])


# Evaluate the anomaly detection using a confusion matrix
print("Evaluating the anomaly detection performance...")
y_true = [0 if label == 1 else 1 for label in y_pred]
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)

tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]

print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")


# Visualize the results using Plotly
print("Visualizing the results using Plotly...")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['creation_time'], y=df['bytes_in'], mode='markers', marker=dict(color=y_pred, colorscale='Viridis')))
fig.add_trace(go.Scatter(x=anomalies['creation_time'], y=anomalies['bytes_in'], mode='markers', marker=dict(color='red', size=10)))
fig.update_layout(title='Anomaly Detection in Network Traffic', xaxis_title='Creation Time', yaxis_title='Bytes In')
fig.show()
