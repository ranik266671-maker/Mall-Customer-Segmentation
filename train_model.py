import pandas as pd
# Mall Customer Segmentation using Machine Learning

# Step 1: Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import joblib

# Step 2: Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 4: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train the Machine Learning model
model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

# Step 6: Assign customer segments
df["Customer_Segment"] = model.labels_

print("\nCustomer Segments:")
print(df[["CustomerID", "Annual Income (k$)",
          "Spending Score (1-100)", "Customer_Segment"]].head(10))

# Step 7: Evaluate the model
score = silhouette_score(X_scaled, model.labels_)
print("\nSilhouette Score:", round(score, 3))

# Step 8: Save the trained model and scaler together
pipeline = Pipeline([
    ("scaler", scaler),
    ("kmeans", model)
])

joblib.dump(pipeline, "customer_segmentation_model.pkl")
print("\nTrained model saved successfully!")

# Step 9: Save results
df.to_csv("customer_segments.csv", index=False)
print("Customer segments saved successfully!")

# Step 10: Visualize customer segments
plt.figure(figsize=(9, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Customer_Segment"],
    cmap="viridis",
    s=60
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-Means")
plt.colorbar(label="Customer Segment")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("customer_segments.png", dpi=150)
plt.show()

# Step 11: Predict the segment for a new customer
new_customer = pd.DataFrame({
    "Annual Income (k$)": [60],
    "Spending Score (1-100)": [50]
})

predicted_segment = pipeline.predict(new_customer)

print("\nNew Customer Details:")
print("Annual Income: 60 thousand dollars")
print("Spending Score: 50")
print("Predicted Customer Segment:", predicted_segment[0])

print("\nProject Completed Successfully!")

# Predict customer segment

import pandas as pd
import joblib

model = joblib.load("customer_segmentation_model.pkl")

new_customer = pd.DataFrame({
    "Annual Income (k$)": [60],
    "Spending Score (1-100)": [50]
})

prediction = model.predict(new_customer)

print("Predicted Customer Segment:", prediction[0])
