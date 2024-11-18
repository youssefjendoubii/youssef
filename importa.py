import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

file_path = 'C:/Users/dalia/Desktop/projetIA/student_data.csv'  
data = pd.read_csv(file_path)


categorical_cols = data.select_dtypes(include=['object']).columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler()),  
            ('pca', PCA(n_components=2))  
        ]), numeric_cols),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', OneHotEncoder(drop='first'))  
        ]), categorical_cols)
    ]
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))  


pipeline.fit(data)


data['Cluster'] = pipeline.named_steps['kmeans'].labels_


sil_score = silhouette_score(
    pipeline.named_steps['preprocessor'].transform(data),
    data['Cluster']
)
print(f'Silhouette Score: {sil_score}')


data_transformed = pipeline.named_steps['preprocessor'].transform(data)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_transformed)


data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

plt.figure(figsize=(10, 8))
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'students wellbeing {cluster}')

plt.title('K-Means Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()


data.to_csv('students_wellbeing_clustered.csv', index=False)

print("Clustered data has been saved to 'students_wellbeing_clustered.csv'.")
