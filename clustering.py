import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Load data 
merged = pd.read_csv('merged.csv')

# unduplicate on SOC code 
merged = merged.drop_duplicates(subset='SOC').reset_index(drop=True)
print(f"Rows after deduplication: {len(merged)}")

#Define feature columns 
state_cols = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
              'Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota',
              'Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
              'New Jersey','New Mexico','New York','North Carolina','North Dakota',
              'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
              'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
              'Virginia','Washington','West Virginia','Wisconsin','Wyoming']

id_cols = ['SOC', 'Occupation', 'O*NET-SOC Code']
drop_cols = state_cols + id_cols
feature_cols = [c for c in merged.columns if c not in drop_cols and c != 'Probability']

X = merged[feature_cols]

#Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized.")

#PCA (reduce to 2 dimensions)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

#  Step 3: Elbow method 
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o', color='steelblue')
plt.title('Elbow Method — Choosing Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('elbow_plot.png')
plt.close()
print("Saved elbow_plot.png")

#K-Means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

#fix fragmentation warning
merged = pd.concat([merged, pd.Series(cluster_labels, name='Cluster')], axis=1)

#Analyze clusters 
print("\nCluster summary (mean automation probability per cluster):")
cluster_summary = merged.groupby('Cluster')['Probability'].agg(['mean', 'count']).round(3)
cluster_summary.columns = ['Mean Probability', 'Count']
cluster_summary = cluster_summary.sort_values('Mean Probability')
print(cluster_summary)

print("\nSample occupations per cluster:")
for c in sorted(merged['Cluster'].unique()):
    sample = merged[merged['Cluster'] == c]['Occupation'].head(5).tolist()
    mean_prob = merged[merged['Cluster'] == c]['Probability'].mean()
    print(f"\nCluster {c} (avg risk: {mean_prob:.2f}):")
    for occ in sample:
        print(f"  - {occ}")

#Step 6: Scatter plot 
colors = ['steelblue', 'crimson', 'seagreen', 'darkorange']

plt.figure(figsize=(10, 7))
for c in range(k):
    mask = merged['Cluster'] == c
    mean_prob = merged[merged['Cluster'] == c]['Probability'].mean()
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[c], label=f"Cluster {c} (avg risk: {mean_prob:.2f})",
        alpha=0.6, edgecolors='white', s=50
    )

plt.title('K-Means Clustering of Occupations by Skill Profile (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('cluster_scatter.png')
plt.close()
print("\nSaved cluster_scatter.png")