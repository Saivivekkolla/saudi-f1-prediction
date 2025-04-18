"""
Model training module for F1 race prediction.
This script trains a machine learning model to predict race positions.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class F1ModelTrainer:
    """Class to train F1 race prediction models."""
    
    def __init__(self, processed_data_dir='../../data/processed', models_dir='../../models'):
        """Initialize the model trainer.
        
        Args:
            processed_data_dir: Directory with processed data files
            models_dir: Directory to store trained models
        """
        self.processed_data_dir = processed_data_dir
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def _load_csv_file(self, filename):
        """Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
    
    def prepare_training_data(self):
        """Prepare data for model training.
        
        Returns:
            X_train, X_test, y_train, y_test: Train-test split data
        """
        print("Preparing training data...")
        
        # Load training data
        training_data = self._load_csv_file('training_data.csv')
        
        if training_data is None or len(training_data) == 0:
            print("No training data found or empty dataset!")
            return None, None, None, None
        
        # Select features and target
        features = [
            'grid_position', 
            'prev_saudi_races', 
            'prev_saudi_best', 
            'prev_saudi_avg'
        ]
        
        X = training_data[features]
        y = training_data['finish_position']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained Random Forest model
        """
        print("Training Random Forest model...")
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create Random Forest regressor
        rf = RandomForestRegressor(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        # Save model
        with open(os.path.join(self.models_dir, 'random_forest_model.pkl'), 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Save best parameters
        best_params = grid_search.best_params_
        with open(os.path.join(self.models_dir, 'random_forest_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"Best Random Forest parameters: {best_params}")
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train):
        """Train an XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost model...")
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create XGBoost regressor
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        
        # Save model
        with open(os.path.join(self.models_dir, 'xgboost_model.pkl'), 'wb') as f:
            pickle.dump(best_xgb, f)
        
        # Save best parameters
        best_params = grid_search.best_params_
        with open(os.path.join(self.models_dir, 'xgboost_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"Best XGBoost parameters: {best_params}")
        
        return best_xgb
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train a Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained Gradient Boosting model
        """
        print("Training Gradient Boosting model...")
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }
        
        # Create Gradient Boosting regressor
        gb = GradientBoostingRegressor(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_gb = grid_search.best_estimator_
        
        # Save model
        with open(os.path.join(self.models_dir, 'gradient_boosting_model.pkl'), 'wb') as f:
            pickle.dump(best_gb, f)
        
        # Save best parameters
        best_params = grid_search.best_params_
        with open(os.path.join(self.models_dir, 'gradient_boosting_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"Best Gradient Boosting parameters: {best_params}")
        
        return best_gb
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate position accuracy
        exact_accuracy = np.mean(np.round(y_pred) == y_test)
        within_1_accuracy = np.mean(np.abs(np.round(y_pred) - y_test) <= 1)
        within_3_accuracy = np.mean(np.abs(np.round(y_pred) - y_test) <= 3)
        
        # Print evaluation results
        print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"{model_name} - Exact position accuracy: {exact_accuracy:.2%}")
        print(f"{model_name} - Within 1 position accuracy: {within_1_accuracy:.2%}")
        print(f"{model_name} - Within 3 positions accuracy: {within_3_accuracy:.2%}")
        
        # Create evaluation metrics dictionary
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'exact_accuracy': exact_accuracy,
            'within_1_accuracy': within_1_accuracy,
            'within_3_accuracy': within_3_accuracy
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.models_dir, f'{model_name}_metrics.csv'), index=False)
        
        # Plot actual vs predicted positions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, 20], [0, 20], 'r--')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title(f'{model_name} - Actual vs Predicted Positions')
        plt.grid(True)
        plt.savefig(os.path.join(self.models_dir, f'{model_name}_predictions.png'))
        plt.close()
        
        return metrics
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models.
        
        Returns:
            Dictionary with trained models and evaluation metrics
        """
        print("Starting model training and evaluation...")
        
        # Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        if X_train is None:
            print("Failed to prepare training data!")
            return None
        
        # Train Random Forest model
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        
        # Train XGBoost model
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
        
        # Train Gradient Boosting model
        gb_model = self.train_gradient_boosting(X_train, y_train)
        gb_metrics = self.evaluate_model(gb_model, X_test, y_test, 'gradient_boosting')
        
        # Compare models
        metrics = [rf_metrics, xgb_metrics, gb_metrics]
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(self.models_dir, 'model_comparison.csv'), index=False)
        
        # Find best model based on MAE
        best_model_idx = metrics_df['mae'].idxmin()
        best_model_name = metrics_df.loc[best_model_idx, 'model_name']
        
        print(f"Best model: {best_model_name}")
        
        # Save best model name
        with open(os.path.join(self.models_dir, 'best_model.txt'), 'w') as f:
            f.write(best_model_name)
        
        # Return models and metrics
        return {
            'random_forest': {
                'model': rf_model,
                'metrics': rf_metrics
            },
            'xgboost': {
                'model': xgb_model,
                'metrics': xgb_metrics
            },
            'gradient_boosting': {
                'model': gb_model,
                'metrics': gb_metrics
            },
            'best_model_name': best_model_name
        }


if __name__ == "__main__":
    trainer = F1ModelTrainer()
    results = trainer.train_and_evaluate_models()
