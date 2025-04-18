"""
Visualization module for F1 race prediction.
This script creates visualizations for the prediction model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class F1Visualizer:
    """Class to create visualizations for F1 predictions."""
    
    def __init__(self, processed_data_dir='../../data/processed', output_dir='../../models'):
        """Initialize the visualizer.
        
        Args:
            processed_data_dir: Directory with processed data files
            output_dir: Directory to store visualizations
        """
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
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
    
    def visualize_driver_performance(self):
        """Visualize driver performance metrics."""
        print("Visualizing driver performance...")
        
        # Load driver features
        driver_features = self._load_csv_file('driver_features.csv')
        
        if driver_features is None:
            print("No driver features data found!")
            return
        
        # Create a figure for driver performance visualization
        plt.figure(figsize=(12, 8))
        
        # Sort drivers by current position
        sorted_drivers = driver_features.sort_values('current_position')
        
        # Plot current points
        sns.barplot(
            x='current_points',
            y='driver_name',
            data=sorted_drivers,
            palette='viridis'
        )
        
        # Set labels and title
        plt.xlabel('Current Season Points')
        plt.ylabel('Driver')
        plt.title('Driver Performance - Current Season Points')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, 'driver_points.png'))
        plt.close()
        
        # Create a figure for Saudi GP specific performance
        plt.figure(figsize=(12, 8))
        
        # Filter drivers with Saudi GP experience
        saudi_experienced = driver_features[driver_features['saudi_races'] > 0].copy()
        
        if len(saudi_experienced) > 0:
            # Sort by average finish position (lower is better)
            saudi_experienced = saudi_experienced.sort_values('saudi_avg_finish')
            
            # Plot average finish position
            sns.barplot(
                x='saudi_avg_finish',
                y='driver_name',
                data=saudi_experienced,
                palette='viridis'
            )
            
            # Set labels and title
            plt.xlabel('Average Finish Position at Saudi Arabian GP')
            plt.ylabel('Driver')
            plt.title('Driver Performance at Saudi Arabian GP')
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(os.path.join(self.output_dir, 'driver_saudi_performance.png'))
        plt.close()
    
    def visualize_constructor_performance(self):
        """Visualize constructor performance metrics."""
        print("Visualizing constructor performance...")
        
        # Load constructor features
        constructor_features = self._load_csv_file('constructor_features.csv')
        
        if constructor_features is None:
            print("No constructor features data found!")
            return
        
        # Create a figure for constructor performance visualization
        plt.figure(figsize=(12, 8))
        
        # Sort constructors by current position
        sorted_constructors = constructor_features.sort_values('current_position')
        
        # Plot current points
        sns.barplot(
            x='current_points',
            y='constructor_name',
            data=sorted_constructors,
            palette='viridis'
        )
        
        # Set labels and title
        plt.xlabel('Current Season Points')
        plt.ylabel('Constructor')
        plt.title('Constructor Performance - Current Season Points')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, 'constructor_points.png'))
        plt.close()
        
        # Create a figure for Saudi GP specific performance
        plt.figure(figsize=(12, 8))
        
        # Filter constructors with Saudi GP experience
        saudi_experienced = constructor_features[constructor_features['saudi_races'] > 0].copy()
        
        if len(saudi_experienced) > 0:
            # Sort by average finish position (lower is better)
            saudi_experienced = saudi_experienced.sort_values('saudi_avg_finish')
            
            # Plot average finish position
            sns.barplot(
                x='saudi_avg_finish',
                y='constructor_name',
                data=saudi_experienced,
                palette='viridis'
            )
            
            # Set labels and title
            plt.xlabel('Average Finish Position at Saudi Arabian GP')
            plt.ylabel('Constructor')
            plt.title('Constructor Performance at Saudi Arabian GP')
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(os.path.join(self.output_dir, 'constructor_saudi_performance.png'))
        plt.close()
    
    def visualize_prediction_factors(self):
        """Visualize factors affecting predictions."""
        print("Visualizing prediction factors...")
        
        # Load combined features
        combined_features = self._load_csv_file('combined_features.csv')
        
        if combined_features is None:
            print("No combined features data found!")
            return
        
        # Create a figure for prediction factors visualization
        plt.figure(figsize=(12, 8))
        
        # Sort by prediction score
        sorted_features = combined_features.sort_values('prediction_score', ascending=False)
        
        # Plot prediction score
        sns.barplot(
            x='prediction_score',
            y='driver_name',
            data=sorted_features,
            palette='viridis'
        )
        
        # Set labels and title
        plt.xlabel('Prediction Score (Higher is Better)')
        plt.ylabel('Driver')
        plt.title('Drivers Ranked by Prediction Score')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, 'prediction_scores.png'))
        plt.close()
        
        # Create a visualization for time-based features
        plt.figure(figsize=(12, 8))
        
        # Prepare data for visualization
        time_features = sorted_features.copy()
        
        # Create a melted dataframe for easier plotting
        time_features_melted = pd.melt(
            time_features,
            id_vars=['driver_name'],
            value_vars=['saudi_recency_factor', 'current_form_factor'],
            var_name='Feature',
            value_name='Value'
        )
        
        # Plot time-based features
        sns.barplot(
            x='Value',
            y='driver_name',
            hue='Feature',
            data=time_features_melted,
            palette='Set2'
        )
        
        # Set labels and title
        plt.xlabel('Feature Value (Higher is Better)')
        plt.ylabel('Driver')
        plt.title('Time-Based Features by Driver')
        plt.legend(title='Feature', loc='lower right')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, 'time_features.png'))
        plt.close()
        
        # Create a correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = combined_features[numeric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5
        )
        
        # Set title
        plt.title('Correlation Between Prediction Factors')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, 'factor_correlations.png'))
        plt.close()
    
    def visualize_all(self):
        """Create all visualizations."""
        print("Creating all visualizations...")
        
        # Visualize driver performance
        self.visualize_driver_performance()
        
        # Visualize constructor performance
        self.visualize_constructor_performance()
        
        # Visualize prediction factors
        self.visualize_prediction_factors()
        
        print("Visualizations completed!")


if __name__ == "__main__":
    visualizer = F1Visualizer()
    visualizer.visualize_all()
