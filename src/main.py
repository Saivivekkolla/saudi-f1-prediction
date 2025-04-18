"""
Main script for Saudi Arabian Grand Prix F1 leaderboard prediction.
This script runs the entire prediction pipeline from data collection to visualization.
"""

import os
import sys
import time
from datetime import datetime

# Set matplotlib backend to non-interactive to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from data.collect_data import F1DataCollector
from data.process_data import F1DataProcessor
from features.build_features import F1FeatureEngineer
from models.train_model import F1ModelTrainer
from models.predict_leaderboard import F1Predictor
from visualization.visualize import F1Visualizer

def run_pipeline():
    """Run the complete prediction pipeline."""
    start_time = time.time()
    
    print("=" * 80)
    print("Saudi Arabian Grand Prix F1 Leaderboard Prediction")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Collect data
    print("\nStep 1: Collecting F1 data...")
    collector = F1DataCollector()
    collector.collect_all_data()
    
    # Step 2: Process data
    print("\nStep 2: Processing F1 data...")
    processor = F1DataProcessor()
    processed_data = processor.process_all_data()
    
    # Step 3: Engineer features
    print("\nStep 3: Engineering features...")
    engineer = F1FeatureEngineer()
    features = engineer.engineer_all_features()
    
    # Step 4: Train models
    print("\nStep 4: Training prediction models...")
    trainer = F1ModelTrainer()
    model_results = trainer.train_and_evaluate_models()
    
    # Step 5: Generate predictions
    print("\nStep 5: Predicting Saudi Arabian GP leaderboard...")
    predictor = F1Predictor()
    leaderboard = predictor.predict()
    
    # Step 6: Create visualizations
    print("\nStep 6: Creating visualizations...")
    visualizer = F1Visualizer()
    visualizer.visualize_all()
    
    # Print execution time
    execution_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Pipeline completed in {execution_time:.2f} seconds")
    print("=" * 80)
    
    # Display the predicted leaderboard
    if leaderboard is not None:
        print("\n2025 Saudi Arabian Grand Prix - Predicted Leaderboard:")
        print("-" * 80)
        print(f"{'Pos':^5} | {'Driver':^25} | {'Lap Time':^15} | {'Starting Grid':^15}")
        print("-" * 80)
        
        for i, row in leaderboard.iterrows():
            pos = int(row['final_position'])
            driver = row['driver_name']
            lap_time = row['lap_time_formatted']
            grid = int(row['grid_position'])
            print(f"{pos:^5} | {driver:^25} | {lap_time:^15} | {grid:^15}")
        
        print("-" * 80)
        print("\nPrediction files saved to data/processed/saudi_gp_leaderboard.csv")
        print("Visualizations saved to models/ directory")
    
    print("=" * 80)
    print("Pipeline execution completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
