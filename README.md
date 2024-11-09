# Ex 10 - Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. The dataset containing customer information, including Annual Income and Spending Score, was loaded for analysis.
2. Relevant features were extracted, and K-Means clustering was applied to the dataset.
3. The elbow method was utilized to determine the optimal number of clusters by plotting WCSS against various cluster counts.
4. After selecting 5 clusters, customers were assigned to the nearest centroid using Euclidean distance.
5. The final clusters were visualized, highlighting distinct customer segments based on spending patterns.

## Program:
```py
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: YUVAN SUNDAR S
RegisterNumber:  212223040250
*/
# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Loading the dataset
file_path = '/mnt/data/Mall_Customers.csv'
data = pd.read_csv(file_path)

# Extracting relevant features: Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Using the elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Applying K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualizing the clusters
plt.figure(figsize=(8,5))
plt.scatter(X[clusters == 0, 0], X[clusters == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[clusters == 1, 0], X[clusters == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[clusters == 2, 0], X[clusters == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[clusters == 3, 0], X[clusters == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[clusters == 4, 0], X[clusters == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
![Screenshot 2024-10-16 090818](https://github.com/user-attachments/assets/5ccaaf96-466f-4e11-b407-480f11a285e2)
![Screenshot 2024-10-16 090828](https://github.com/user-attachments/assets/5b147c8b-633e-44b5-b07c-56cd5036afe3)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
