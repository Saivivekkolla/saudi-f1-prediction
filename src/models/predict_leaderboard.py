"""
Prediction module for F1 race prediction.
This script predicts the leaderboard for the Saudi Arabian Grand Prix.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class F1Predictor:
    """Class to predict F1 race results."""
    
    def __init__(self, processed_data_dir='../../data/processed', models_dir='../../models'):
        """Initialize the predictor.
        
        Args:
            processed_data_dir: Directory with processed data files
            models_dir: Directory with trained models
        """
        self.processed_data_dir = processed_data_dir
        self.models_dir = models_dir
    
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
    
    def _load_model(self, model_name):
        """Load a trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _get_best_model_name(self):
        """Get the name of the best model.
        
        Returns:
            Name of the best model
        """
        best_model_path = os.path.join(self.models_dir, 'best_model.txt')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                return f.read().strip()
        return 'xgboost'  # Default to XGBoost if no best model is specified
    
    def prepare_prediction_data(self):
        """Prepare data for prediction.
        
        Returns:
            DataFrame with prediction data
        """
        print("Preparing prediction data...")
        
        # Load combined features
        combined_features = self._load_csv_file('combined_features.csv')
        
        if combined_features is None:
            print("No combined features data found!")
            return None
        
        # Load drivers data
        drivers = self._load_csv_file('drivers_2024.csv')
        
        if drivers is None:
            print("No drivers data found!")
            return None
        
        # Create prediction data
        prediction_data = []
        
        for _, driver in combined_features.iterrows():
            # Get driver info
            driver_id = driver['driver_id']
            driver_name = driver['driver_name']
            
            # Estimate grid position based on current standings
            # This is a simplification; in reality, qualifying would determine this
            estimated_grid = driver['current_position_driver'] if not pd.isna(driver['current_position_driver']) else 20
            
            # Use previous Saudi GP performance if available
            prev_saudi_races = driver['saudi_races_driver'] if not pd.isna(driver['saudi_races_driver']) else 0
            prev_saudi_best = driver['saudi_best_finish_driver'] if not pd.isna(driver['saudi_best_finish_driver']) else 20
            prev_saudi_avg = driver['saudi_avg_finish_driver'] if not pd.isna(driver['saudi_avg_finish_driver']) else 20
            
            # Create prediction entry
            prediction_entry = {
                'driver_id': driver_id,
                'driver_name': driver_name,
                'grid_position': estimated_grid,
                'prev_saudi_races': prev_saudi_races,
                'prev_saudi_best': prev_saudi_best,
                'prev_saudi_avg': prev_saudi_avg,
                'prediction_score': driver['prediction_score'] if not pd.isna(driver['prediction_score']) else 0,
                'saudi_recency_factor': driver['saudi_recency_factor'] if not pd.isna(driver['saudi_recency_factor']) else 0,
                'current_form_factor': driver['current_form_factor'] if not pd.isna(driver['current_form_factor']) else 0
            }
            
            prediction_data.append(prediction_entry)
        
        # Convert to DataFrame
        prediction_df = pd.DataFrame(prediction_data)
        
        # Save prediction data
        prediction_df.to_csv(os.path.join(self.processed_data_dir, 'prediction_data.csv'), index=False)
        
        return prediction_df
    
    def predict_leaderboard(self):
        """Predict the leaderboard for the Saudi Arabian Grand Prix.
        
        Returns:
            DataFrame with predicted leaderboard
        """
        print("Predicting Saudi Arabian GP leaderboard...")
        
        # Prepare prediction data
        prediction_data = self.prepare_prediction_data()
        
        if prediction_data is None:
            print("Failed to prepare prediction data!")
            return None
        
        # Get best model name
        best_model_name = self._get_best_model_name()
        print(f"Using {best_model_name} for prediction...")
        
        # Load best model
        model = self._load_model(best_model_name)
        
        if model is None:
            print(f"Failed to load {best_model_name} model!")
            return None
        
        # Select features for prediction - use only the features that were available during training
        features = [
            'grid_position', 
            'prev_saudi_races', 
            'prev_saudi_best', 
            'prev_saudi_avg'
        ]
        
        # Make predictions
        X_pred = prediction_data[features]
        predicted_positions = model.predict(X_pred)
        
        # Add predictions to DataFrame
        prediction_data['predicted_position'] = predicted_positions
        
        # Generate estimated lap times based on predicted positions
        # Jeddah Corniche Circuit reference lap time (in seconds)
        reference_lap_time = 90.0  # ~1:30.000 as base reference time
        
        # Calculate estimated lap times (faster positions get faster times with diminishing returns)
        prediction_data['estimated_lap_time'] = reference_lap_time + (np.log1p(prediction_data['predicted_position']) * 0.8)
        
        # Add some randomness to lap times (within 0.3 seconds)
        np.random.seed(42)  # For reproducibility
        prediction_data['estimated_lap_time'] += np.random.uniform(-0.3, 0.3, size=len(prediction_data))
        
        # Format lap times as MM:SS.mmm
        prediction_data['lap_time_formatted'] = prediction_data['estimated_lap_time'].apply(
            lambda x: f"{int(x // 60):01d}:{x % 60:06.3f}"
        )
        
        # Sort by predicted position
        leaderboard = prediction_data.sort_values('predicted_position')
        
        # Add final position (1st, 2nd, 3rd, etc.)
        leaderboard['final_position'] = range(1, len(leaderboard) + 1)
        
        # Reorder columns for better readability
        leaderboard = leaderboard[[
            'final_position',
            'driver_name',
            'predicted_position',
            'lap_time_formatted',
            'estimated_lap_time',  
            'grid_position',
            'prediction_score',
            'saudi_recency_factor',
            'current_form_factor',
            'driver_id',
            'prev_saudi_races',
            'prev_saudi_best',
            'prev_saudi_avg'
        ]]
        
        # Save leaderboard
        leaderboard.to_csv(os.path.join(self.processed_data_dir, 'saudi_gp_leaderboard.csv'), index=False)
        
        # Create a simplified leaderboard for display
        display_leaderboard = leaderboard[['final_position', 'driver_name', 'grid_position']].copy()
        display_leaderboard.columns = ['Position', 'Driver', 'Starting Grid']
        
        # Save display leaderboard
        display_leaderboard.to_csv(os.path.join(self.processed_data_dir, 'saudi_gp_display_leaderboard.csv'), index=False)
        
        return leaderboard
    
    def visualize_leaderboard(self, leaderboard):
        """Visualize the predicted leaderboard.
        
        Args:
            leaderboard: DataFrame with predicted leaderboard
        """
        print("Visualizing leaderboard...")
        
        # Create a figure for the leaderboard visualization
        plt.figure(figsize=(12, 8))
        
        # Plot the leaderboard
        sns.barplot(
            x='final_position',
            y='driver_name',
            data=leaderboard.sort_values('final_position'),
            palette='viridis'
        )
        
        # Add grid position annotations
        for i, row in leaderboard.sort_values('final_position').iterrows():
            plt.text(
                0.1,
                i,
                f"Grid: {int(row['grid_position'])}",
                va='center',
                fontsize=8,
                color='white'
            )
        
        # Set labels and title
        plt.xlabel('Predicted Finishing Position')
        plt.ylabel('Driver')
        plt.title('2025 Saudi Arabian Grand Prix - Predicted Leaderboard')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.models_dir, 'saudi_gp_leaderboard.png'))
        plt.close()
        
        # Create a lap time visualization
        plt.figure(figsize=(12, 8))
        
        # Extract lap times in seconds for plotting
        lap_times = leaderboard.copy()
        
        # Sort by lap time (fastest first)
        lap_times = lap_times.sort_values('estimated_lap_time')
        
        # Create color gradient based on lap times (faster = greener)
        norm = plt.Normalize(lap_times['estimated_lap_time'].min(), lap_times['estimated_lap_time'].max())
        colors = plt.cm.RdYlGn_r(norm(lap_times['estimated_lap_time']))
        
        # Plot lap times
        bars = plt.barh(
            lap_times['driver_name'],
            lap_times['estimated_lap_time'],
            color=colors
        )
        
        # Add lap time annotations
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1,
                i,
                lap_times.iloc[i]['lap_time_formatted'],
                va='center',
                fontsize=9
            )
        
        # Set labels and title
        plt.xlabel('Lap Time (seconds)')
        plt.ylabel('Driver')
        plt.title('2025 Saudi Arabian Grand Prix - Predicted Lap Times')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.models_dir, 'saudi_gp_lap_times.png'))
        plt.close()
        
        # Create a grid vs finish position plot
        plt.figure(figsize=(10, 6))
        
        # Plot grid vs predicted position
        plt.scatter(
            leaderboard['grid_position'],
            leaderboard['final_position'],
            alpha=0.7,
            s=100
        )
        
        # Add driver labels
        for i, row in leaderboard.iterrows():
            plt.annotate(
                row['driver_name'].split()[-1],  # Last name only
                (row['grid_position'], row['final_position']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
        
        # Add diagonal line (grid = finish)
        plt.plot([0, 20], [0, 20], 'r--', alpha=0.5)
        
        # Set labels and title
        plt.xlabel('Grid Position')
        plt.ylabel('Predicted Finishing Position')
        plt.title('Grid Position vs Predicted Finishing Position')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.models_dir, 'saudi_gp_grid_vs_finish.png'))
        plt.close()
    
    def predict(self):
        """Run the full prediction pipeline."""
        print("Starting prediction pipeline...")
        
        # Predict leaderboard
        leaderboard = self.predict_leaderboard()
        
        if leaderboard is not None:
            # Visualize leaderboard
            self.visualize_leaderboard(leaderboard)
            
            # Display top 10 finishers
            top10 = leaderboard.head(10)
            print("\n2025 Saudi Arabian Grand Prix - Predicted Top 10:")
            for i, row in top10.iterrows():
                print(f"{int(row['final_position'])}. {row['driver_name']}")
        
        print("Prediction completed!")
        
        return leaderboard


if __name__ == "__main__":
    predictor = F1Predictor()
    leaderboard = predictor.predict()
