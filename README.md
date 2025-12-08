# ESG Dataset Clustering & Prediction Pipeline

A comprehensive pipeline for ESG (Environmental, Social, Governance) datasets featuring:
- **Unsupervised Learning**: Clustering using K-Means, Gaussian Mixture Models (GMM), and HDBSCAN
- **Supervised Learning**: ESG Score prediction using Random Forest and LightGBM regressors

## 🎯 Project Overview

This project provides a complete end-to-end solution for ESG data analysis with:

### Unsupervised Learning (Clustering)
- **Data Preprocessing**: Missing value handling, feature scaling, and PCA for visualization
- **Multiple Clustering Algorithms**: K-Means, GMM, and HDBSCAN
- **Hyperparameter Optimization**: Automatic optimization for all three algorithms
- **Model Evaluation**: Comprehensive comparison using multiple metrics
- **Cluster Interpretation**: Automatic profiling and interpretation of clusters

### Supervised Learning (Prediction)
- **ESG Score Prediction**: Predict ESG scores using cluster-enriched features
- **Multiple Models**: Random Forest and LightGBM regressors
- **Hyperparameter Optimization**: Optuna-based optimization with cross-validation
- **Comprehensive Metrics**: R², RMSE, MAE, MAPE
- **Feature Importance Analysis**: Understand which features drive predictions
- **Model Comparison**: Side-by-side comparison with visualizations

### Interactive Dashboard
- **Streamlit-based UI**: Easy-to-use interface for both clustering and prediction
- **Real-time Visualizations**: Interactive plots and charts
- **Model Download**: Save trained models for production use

## 📋 Features

### Preprocessing
- ✅ Automatic exclusion of `Company_ID` and `Sector` features
- ✅ Missing value imputation (mean imputation)
- ✅ Feature scaling using StandardScaler
- ✅ PCA for 2D/3D visualization

### Clustering Algorithms
- ✅ **K-Means**: Optimized n_clusters parameter
- ✅ **GMM**: Optimized n_components and covariance_type
- ✅ **HDBSCAN**: Optimized min_cluster_size and min_samples

### Evaluation Metrics
- ✅ Silhouette Score (higher is better)
- ✅ Davies-Bouldin Index (lower is better)
- ✅ Calinski-Harabasz Score (higher is better)
- ✅ Automatic best model selection

### Cluster Interpretation
- ✅ Cluster profiling based on mean feature values
- ✅ Top and bottom features identification
- ✅ Cluster size statistics
- ✅ Heuristic-based cluster interpretation

### Streamlit Dashboard
- ✅ Dataset preview and statistics
- ✅ Preprocessing summary
- ✅ Algorithm selection and execution
- ✅ Manual hyperparameter tuning
- ✅ Automatic hyperparameter optimization
- ✅ **Elbow Method** for optimal K-Means cluster selection
- ✅ Model comparison table
- ✅ Interactive 2D/3D PCA visualizations
- ✅ Cluster profiling and interpretation
- ✅ Downloadable results (CSV)

### ESG Score Prediction
- ✅ **Random Forest Regressor**: Ensemble method with feature importance
- ✅ **LightGBM Regressor**: Gradient boosting with fast training
- ✅ **Optuna Optimization**: Automated hyperparameter tuning
- ✅ **Cross-Validation**: K-Fold CV for robust evaluation
- ✅ **Comprehensive Metrics**: R², RMSE, MAE, MAPE
- ✅ **Feature Importance**: Visualize which features matter most
- ✅ **Prediction vs True Values**: Scatter plots and residual analysis
- ✅ **New Data Prediction**: Input form for predicting new ESG scores
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Cluster-Enriched Features**: Use cluster labels as additional features

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your dataset is in the correct location**:
   - Place your `esg_dataset.csv` file in the `data/` directory
   - The dataset should have columns: `Company_ID`, `Sector`, and various ESG metrics

## 📖 Usage

### Running the Streamlit App

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **The app will open in your browser** (usually at `http://localhost:8501`)

3. **Follow these steps in the app**:
   
   **For Clustering:**
   - **Load Data**: Click "Load & Preprocess Data" in the sidebar
   - **Elbow Method** (optional): Use the "Méthode du Coude" section to find optimal K for K-Means
   - **Choose Algorithms**: Select which clustering algorithms to use
   - **Set Parameters**: Either use manual parameters or run auto-optimization
   - **Run Clustering**: Click "Run Clustering" to execute
   - **View Results**: Explore results in the "Results" and "Visualizations" tabs
   - **Download**: Download the labeled dataset as CSV
   
   **For Prediction:**
   - **Complete Clustering First**: Run clustering and download the results to create `data/esg_clustered_results.csv`
   - **Go to Prediction Tab**: Navigate to "🔮 ESG Score Prediction" tab
   - **Note**: The prediction module uses `data/esg_clustered_results.csv` which includes cluster labels
   - **Select Models**: Choose Random Forest and/or LightGBM
   - **Configure Training**: Set optimization trials, CV folds, and test split
   - **Include Cluster Labels**: Optionally include the Cluster column as a feature
   - **Train Models**: Click "🚀 Train Regression Models"
   - **View Results**: Explore metrics, visualizations, and feature importance
   - **Predict New Scores**: Use the input form to predict ESG scores for new data
   - **Download Model**: Save the best model for production use

### Using the Modules Programmatically

You can also use the modules directly in Python:

```python
from preprocessing import DataPreprocessor
from clustering import ClusteringModels
from optimization import HyperparameterOptimizer
from evaluation import ClusteringEvaluator
from labeling import ClusterProfiler

# Load and preprocess data
preprocessor = DataPreprocessor("data/esg_dataset.csv")
data_scaled, pca_2d, pca_3d = preprocessor.preprocess_pipeline()

# Optimize hyperparameters
optimizer = HyperparameterOptimizer()
kmeans_params = optimizer.optimize_kmeans(data_scaled)
gmm_params = optimizer.optimize_gmm(data_scaled)
hdbscan_params = optimizer.optimize_hdbscan(data_scaled)

# Train models with optimal parameters
models = ClusteringModels()
labels_kmeans = models.fit_kmeans(data_scaled, n_clusters=kmeans_params['n_clusters'])
labels_gmm = models.fit_gmm(data_scaled, n_components=gmm_params['n_components'],
                           covariance_type=gmm_params['covariance_type'])
labels_hdbscan = models.fit_hdbscan(data_scaled, 
                                    min_cluster_size=hdbscan_params['min_cluster_size'],
                                    min_samples=hdbscan_params['min_samples'])

# Evaluate models
evaluator = ClusteringEvaluator()
comparison_df = evaluator.compare_models(data_scaled, models, {
    'kmeans': labels_kmeans,
    'gmm': labels_gmm,
    'hdbscan': labels_hdbscan
})

# Get best model
best_model = evaluator.select_best_model(comparison_df)

# Profile clusters
profiler = ClusterProfiler(preprocessor.feature_names)
profiles = profiler.profile_clusters(data_scaled, labels_kmeans)
interpretations = profiler.get_cluster_interpretations(data_scaled, labels_kmeans)
```

## 📁 Project Structure

```
esgprojetfinal/
├── app.py                      # Streamlit dashboard
├── preprocessing.py            # Data loading and preprocessing
├── clustering.py               # Clustering algorithms (KMeans, GMM, HDBSCAN)
├── optimization.py             # Hyperparameter optimization (clustering)
├── evaluation.py               # Clustering evaluation and comparison
├── labeling.py                 # Cluster profiling and interpretation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── prediction/                 # Supervised learning models
│   ├── __init__.py
│   ├── regression_random_forest.py    # Random Forest regressor
│   └── regression_lightgbm.py        # LightGBM regressor
├── training/                   # Training utilities
│   ├── __init__.py
│   └── train_regressors.py    # Model training pipeline
├── evaluation/                 # Regression evaluation
│   ├── __init__.py
│   └── regression_metrics.py  # R², RMSE, MAE, MAPE metrics
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── model_selection.py      # Model comparison utilities
├── models/                     # Saved models
│   └── best_model.pkl         # Best trained model (generated)
└── data/
    └── esg_dataset.csv        # Your ESG dataset
```

## 🔧 Configuration

### Dataset Format

Your `esg_dataset.csv` should have:
- `Company_ID`: Unique identifier for each company (will be excluded from clustering)
- `Sector`: Company sector (will be excluded from clustering)
- ESG features: Numerical features for clustering (e.g., CO2_Emissions, Energy_Consumption, etc.)

### Hyperparameter Ranges

Default optimization ranges:
- **K-Means**: n_clusters from 2 to min(10, n_samples/2)
- **GMM**: n_components from 2 to min(10, n_samples/2), covariance_type: ['full', 'tied', 'diag', 'spherical']
- **HDBSCAN**: min_cluster_size: [3, 5, 7, 10, 15], min_samples: [3, 5, 7, 10]

You can modify these in the `optimization.py` file or use manual parameters in the Streamlit UI.

## 📊 Output

### Clustering Output
1. **Preprocessed dataset**: Scaled and cleaned data
2. **Cluster labels**: Assigned cluster for each sample
3. **Model comparison**: Metrics for all three algorithms
4. **Best model selection**: Automatically selected optimal model
5. **Cluster profiles**: Mean feature values per cluster
6. **Cluster interpretations**: Heuristic-based descriptions
7. **Visualizations**: 2D/3D PCA plots colored by clusters
8. **Labeled dataset**: CSV file with original data + cluster labels

### Prediction Output
1. **Trained models**: Random Forest and/or LightGBM regressors
2. **Performance metrics**: R², RMSE, MAE, MAPE (test and CV)
3. **Model comparison**: Side-by-side comparison DataFrame
4. **Visualizations**: 
   - Prediction vs True values scatter plots
   - Residuals distribution
   - Feature importance plots
   - Radar charts and bar comparisons
5. **Best model**: Automatically selected based on R² score
6. **Saved model**: Pickle file of the best model (`models/best_model.pkl`)
7. **Predictions CSV**: Test set predictions with residuals

## 🎓 Understanding the Metrics

### Clustering Metrics
- **Silhouette Score**: Measures how similar an object is to its own cluster vs. other clusters. Range: -1 to 1 (higher is better)
- **Davies-Bouldin Index**: Average similarity ratio of clusters. Lower is better
- **Calinski-Harabasz Score**: Ratio of between-clusters to within-cluster dispersion. Higher is better

### Regression Metrics
- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model. Range: -∞ to 1 (higher is better, 1 = perfect prediction)
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors. Lower is better, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true values. Lower is better, in same units as target
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Lower is better, expressed as percentage

### 📊 Elbow Method

The **Elbow Method** is a heuristic technique to determine the optimal number of clusters for K-Means. It works by:

1. **Calculating inertia** (within-cluster sum of squares) for different values of K
2. **Plotting inertia vs. K**: As K increases, inertia decreases. The "elbow" point is where the rate of decrease slows down significantly
3. **Finding the optimal K**: The elbow point suggests the optimal number of clusters

**How to use in the app**:
- Go to the "Clustering" tab
- Use the "Méthode du Coude" section
- Set the maximum number of clusters to test
- Click "Calculer" to compute and visualize the elbow method
- The app will suggest an optimal K value based on the elbow point and silhouette scores
- Use this K value in the K-Means parameters

### 📌 Understanding HDBSCAN Labels

**Important**: HDBSCAN uses the label **-1** to mark noise points (outliers) that don't belong to any cluster. This is **not** a cluster, but rather points that couldn't be assigned to any cluster.

**Example**: If you see labels `-1, 0, 1, 2`, this means:
- **3 clusters**: labels 0, 1, and 2
- **Noise points**: label -1 (outliers)

The number of clusters reported excludes the -1 label. So if HDBSCAN finds 3 clusters, you'll see labels 0, 1, 2 for the clusters, and potentially -1 for noise points.

### 🔧 Reducing Noise in HDBSCAN

If you have **too many noise points** (>15-20% of your dataset), try:

1. **Reduce `min_cluster_size`**: Lower values (2-3) allow smaller clusters and reduce noise
   - For datasets < 200 points: use 2-3
   - For datasets 200-500 points: use 3-5
   - For larger datasets: use 5-10

2. **Reduce `min_samples`**: Lower values (2-3) make the algorithm more permissive
   - For datasets < 200 points: use 2-3
   - For larger datasets: use 3-5

3. **Reassign noise points**: Use the "Réassigner les points de bruit" option in the Streamlit UI to automatically assign noise points to the nearest cluster

4. **Adjust optimization**: The optimization now automatically penalizes high noise ratios and uses smaller default values for smaller datasets

## 🐛 Troubleshooting

### Common Issues

**Clustering:**
1. **FileNotFoundError**: Make sure `esg_dataset.csv` is in the `data/` directory
2. **Memory Error**: For large datasets, consider reducing the hyperparameter search space
3. **No clusters found (HDBSCAN)**: Try reducing `min_cluster_size` or `min_samples`
4. **All points in one cluster**: This usually indicates the data needs different preprocessing or the number of clusters is too high
5. **Too much noise in HDBSCAN (>30% of points)**: 
   - Reduce `min_cluster_size` to 2-3
   - Reduce `min_samples` to 2-3
   - Use the "Réassigner les points de bruit" option in the UI
   - The optimization now automatically favors configurations with less noise

**Prediction:**
1. **ImportError for LightGBM**: Install with `pip install lightgbm`
2. **ImportError for Optuna**: Install with `pip install optuna`
3. **Target column not found**: Ensure your dataset has the target column (default: `ESG_Score`)
4. **Model training fails**: 
   - Ensure clustering has been completed first (for cluster-enriched features)
   - Check that the dataset has sufficient samples (minimum 20-30 for meaningful training)
   - Reduce optimization trials if training is too slow
5. **Poor prediction performance**:
   - Try including/excluding cluster labels as features
   - Increase optimization trials for better hyperparameters
   - Check feature importance to identify relevant features
   - Ensure target variable has sufficient variance

### Performance Tips

- For large datasets (>10,000 samples), consider sampling before optimization
- HDBSCAN can be slow on large datasets; consider using approximate methods
- PCA visualization works best with 2-3 components

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Clustering! 🎉**
