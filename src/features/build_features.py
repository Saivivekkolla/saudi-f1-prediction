"""
Feature engineering module for F1 race prediction.
This script creates features for the machine learning model.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

class F1FeatureEngineer:
    """Class to engineer features for the F1 prediction model."""
    
    def __init__(self, processed_data_dir='../../data/processed', output_dir='../../data/processed'):
        """Initialize the feature engineer.
        
        Args:
            processed_data_dir: Directory with processed data files
            output_dir: Directory to store feature files
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
    
    def engineer_driver_features(self):
        """Engineer driver-related features.
        
        Returns:
            DataFrame with driver features
        """
        print("Engineering driver features...")
        
        # Load processed data
        race_results = self._load_csv_file('race_results.csv')
        qualifying_results = self._load_csv_file('qualifying_results.csv')
        driver_standings = self._load_csv_file('driver_standings_2024.csv')
        drivers = self._load_csv_file('drivers_2024.csv')
        
        if race_results is None or qualifying_results is None or driver_standings is None or drivers is None:
            print("Missing required data files!")
            return None
        
        # Create driver features DataFrame
        driver_features = []
        
        # Get unique drivers from 2024 data
        current_drivers = drivers['driver_id'].unique()
        
        for driver_id in current_drivers:
            # Get driver info
            driver_info = drivers[drivers['driver_id'] == driver_id].iloc[0]
            driver_name = driver_info['driver_name']
            
            # Get driver standings
            driver_standing = driver_standings[driver_standings['driver_id'] == driver_id]
            current_position = driver_standing['position'].iloc[0] if not driver_standing.empty else None
            current_points = driver_standing['points'].iloc[0] if not driver_standing.empty else 0
            current_wins = driver_standing['wins'].iloc[0] if not driver_standing.empty else 0
            
            # Get constructor ID
            constructor_id = driver_standing['constructor_id'].iloc[0] if not driver_standing.empty else None
            
            # Calculate Saudi Arabian GP specific stats
            driver_saudi_results = race_results[race_results['driver_id'] == driver_id]
            
            # Past performance at Saudi Arabian GP
            saudi_races = len(driver_saudi_results)
            saudi_best_finish = driver_saudi_results['finish_position'].min() if not driver_saudi_results.empty else None
            saudi_avg_finish = driver_saudi_results['finish_position'].mean() if not driver_saudi_results.empty else None
            saudi_wins = len(driver_saudi_results[driver_saudi_results['finish_position'] == 1])
            saudi_podiums = len(driver_saudi_results[driver_saudi_results['finish_position'] <= 3])
            
            # Qualifying performance at Saudi Arabian GP
            driver_saudi_quali = qualifying_results[qualifying_results['driver_id'] == driver_id]
            saudi_best_quali = driver_saudi_quali['position'].min() if not driver_saudi_quali.empty else None
            saudi_avg_quali = driver_saudi_quali['position'].mean() if not driver_saudi_quali.empty else None
            saudi_poles = len(driver_saudi_quali[driver_saudi_quali['position'] == 1])
            
            # Calculate overall race stats
            total_races = len(race_results[race_results['driver_id'] == driver_id])
            
            # Create feature entry
            feature_entry = {
                'driver_id': driver_id,
                'driver_name': driver_name,
                'constructor_id': constructor_id,
                'current_position': current_position,
                'current_points': current_points,
                'current_wins': current_wins,
                'saudi_races': saudi_races,
                'saudi_best_finish': saudi_best_finish,
                'saudi_avg_finish': saudi_avg_finish,
                'saudi_wins': saudi_wins,
                'saudi_podiums': saudi_podiums,
                'saudi_best_quali': saudi_best_quali,
                'saudi_avg_quali': saudi_avg_quali,
                'saudi_poles': saudi_poles,
                'total_races': total_races
            }
            
            driver_features.append(feature_entry)
        
        # Convert to DataFrame
        driver_features_df = pd.DataFrame(driver_features)
        
        # Fill missing values
        driver_features_df['saudi_races'] = driver_features_df['saudi_races'].fillna(0)
        driver_features_df['saudi_wins'] = driver_features_df['saudi_wins'].fillna(0)
        driver_features_df['saudi_podiums'] = driver_features_df['saudi_podiums'].fillna(0)
        driver_features_df['saudi_poles'] = driver_features_df['saudi_poles'].fillna(0)
        
        # Calculate experience factor (normalized)
        max_races = driver_features_df['total_races'].max()
        driver_features_df['experience_factor'] = driver_features_df['total_races'] / max_races if max_races > 0 else 0
        
        # Save engineered features
        driver_features_df.to_csv(os.path.join(self.output_dir, 'driver_features.csv'), index=False)
        
        return driver_features_df
    
    def engineer_constructor_features(self):
        """Engineer constructor-related features.
        
        Returns:
            DataFrame with constructor features
        """
        print("Engineering constructor features...")
        
        # Load processed data
        race_results = self._load_csv_file('race_results.csv')
        constructor_standings = self._load_csv_file('constructor_standings_2024.csv')
        constructors = self._load_csv_file('constructors_2024.csv')
        
        if race_results is None or constructor_standings is None or constructors is None:
            print("Missing required data files!")
            return None
        
        # Create constructor features DataFrame
        constructor_features = []
        
        # Get unique constructors from 2024 data
        current_constructors = constructors['constructor_id'].unique()
        
        for constructor_id in current_constructors:
            # Get constructor info
            constructor_info = constructors[constructors['constructor_id'] == constructor_id].iloc[0]
            constructor_name = constructor_info['constructor_name']
            
            # Get constructor standings
            constructor_standing = constructor_standings[constructor_standings['constructor_id'] == constructor_id]
            current_position = constructor_standing['position'].iloc[0] if not constructor_standing.empty else None
            current_points = constructor_standing['points'].iloc[0] if not constructor_standing.empty else 0
            current_wins = constructor_standing['wins'].iloc[0] if not constructor_standing.empty else 0
            
            # Calculate Saudi Arabian GP specific stats
            constructor_saudi_results = race_results[race_results['constructor_id'] == constructor_id]
            
            # Past performance at Saudi Arabian GP
            saudi_races = len(constructor_saudi_results.drop_duplicates(subset=['year']))
            saudi_best_finish = constructor_saudi_results['finish_position'].min() if not constructor_saudi_results.empty else None
            saudi_avg_finish = constructor_saudi_results['finish_position'].mean() if not constructor_saudi_results.empty else None
            saudi_wins = len(constructor_saudi_results[constructor_saudi_results['finish_position'] == 1])
            saudi_podiums = len(constructor_saudi_results[constructor_saudi_results['finish_position'] <= 3])
            
            # Create feature entry
            feature_entry = {
                'constructor_id': constructor_id,
                'constructor_name': constructor_name,
                'current_position': current_position,
                'current_points': current_points,
                'current_wins': current_wins,
                'saudi_races': saudi_races,
                'saudi_best_finish': saudi_best_finish,
                'saudi_avg_finish': saudi_avg_finish,
                'saudi_wins': saudi_wins,
                'saudi_podiums': saudi_podiums
            }
            
            constructor_features.append(feature_entry)
        
        # Convert to DataFrame
        constructor_features_df = pd.DataFrame(constructor_features)
        
        # Fill missing values
        constructor_features_df['saudi_races'] = constructor_features_df['saudi_races'].fillna(0)
        constructor_features_df['saudi_wins'] = constructor_features_df['saudi_wins'].fillna(0)
        constructor_features_df['saudi_podiums'] = constructor_features_df['saudi_podiums'].fillna(0)
        
        # Save engineered features
        constructor_features_df.to_csv(os.path.join(self.output_dir, 'constructor_features.csv'), index=False)
        
        return constructor_features_df
    
    def engineer_combined_features(self):
        """Engineer combined features for model training.
        
        Returns:
            DataFrame with combined features
        """
        print("Engineering combined features...")
        
        # Load engineered features
        driver_features = self._load_csv_file('driver_features.csv')
        constructor_features = self._load_csv_file('constructor_features.csv')
        
        if driver_features is None or constructor_features is None:
            print("Missing engineered feature files!")
            return None
        
        # Merge driver and constructor features
        combined_features = pd.merge(
            driver_features,
            constructor_features,
            on='constructor_id',
            how='left',
            suffixes=('_driver', '_constructor')
        )
        
        # Engineer additional combined features
        
        # Driver-Constructor synergy (higher is better)
        combined_features['driver_constructor_synergy'] = (
            (combined_features['current_points_driver'] / combined_features['current_points_driver'].max() if combined_features['current_points_driver'].max() > 0 else 0) +
            (combined_features['current_points_constructor'] / combined_features['current_points_constructor'].max() if combined_features['current_points_constructor'].max() > 0 else 0)
        ) / 2
        
        # Saudi GP performance score (higher is better)
        combined_features['saudi_performance_score'] = np.nan
        
        # Calculate Saudi performance score only for drivers with previous Saudi GP experience
        mask = combined_features['saudi_races_driver'] > 0
        if mask.any():
            combined_features.loc[mask, 'saudi_performance_score'] = (
                (1 / combined_features.loc[mask, 'saudi_avg_finish_driver']) * 0.5 +
                (combined_features.loc[mask, 'saudi_podiums_driver'] / combined_features.loc[mask, 'saudi_races_driver']) * 0.3 +
                (combined_features.loc[mask, 'saudi_poles'] / combined_features.loc[mask, 'saudi_races_driver']) * 0.2
            )
        
        # For drivers without Saudi experience, use current season performance
        mask = combined_features['saudi_races_driver'] == 0
        if mask.any():
            max_points = combined_features['current_points_driver'].max()
            if max_points > 0:
                combined_features.loc[mask, 'saudi_performance_score'] = (
                    combined_features.loc[mask, 'current_points_driver'] / max_points
                )
            else:
                combined_features.loc[mask, 'saudi_performance_score'] = 0
        
        # Add time-based features
        # Current date for the 2025 Saudi Arabian GP (April 18-20, 2025)
        current_date = datetime(2025, 4, 18)
        
        # Calculate recency factor for Saudi GP experience (more recent experience is better)
        # Get the latest Saudi GP date for each driver
        race_results = self._load_csv_file('race_results.csv')
        if race_results is not None:
            # Convert race_date to datetime
            race_results['race_date'] = pd.to_datetime(race_results['race_date'])
            
            # Calculate recency factor for each driver
            for i, row in combined_features.iterrows():
                driver_id = row['driver_id']
                
                # Get driver's Saudi GP results
                driver_results = race_results[race_results['driver_id'] == driver_id]
                
                if not driver_results.empty:
                    # Get the most recent Saudi GP date for this driver
                    latest_saudi_date = driver_results['race_date'].max()
                    
                    # Calculate days since last Saudi GP
                    days_since_last_race = (current_date - latest_saudi_date).days
                    
                    # Calculate recency factor (exponential decay - more recent is better)
                    # Scale to be between 0 and 1, with 1 being most recent
                    recency_factor = np.exp(-days_since_last_race / 365)  # Decay over a year
                    
                    # Add recency factor
                    combined_features.loc[i, 'saudi_recency_factor'] = recency_factor
                else:
                    # No Saudi GP experience
                    combined_features.loc[i, 'saudi_recency_factor'] = 0
        
        # Fill missing values
        combined_features['saudi_recency_factor'] = combined_features['saudi_recency_factor'].fillna(0)
        
        # Add seasonal performance trend (recent form)
        # Higher weight for more recent races in the current season
        driver_standings = self._load_csv_file('driver_standings_2024.csv')
        if driver_standings is not None:
            # Calculate form factor based on current standings and points
            max_points = driver_standings['points'].max()
            if max_points > 0:
                for i, row in combined_features.iterrows():
                    driver_id = row['driver_id']
                    
                    # Get driver's current standings
                    driver_standing = driver_standings[driver_standings['driver_id'] == driver_id]
                    
                    if not driver_standing.empty:
                        # Calculate form factor (normalized points with position bonus)
                        points = driver_standing['points'].iloc[0]
                        position = driver_standing['position'].iloc[0]
                        
                        # Points normalized to 0-1 scale
                        points_factor = points / max_points
                        
                        # Position bonus (1st = 1.0, 20th = 0.0)
                        position_factor = (21 - position) / 20
                        
                        # Combined form factor
                        form_factor = (points_factor * 0.7) + (position_factor * 0.3)
                        
                        # Add form factor
                        combined_features.loc[i, 'current_form_factor'] = form_factor
                    else:
                        # No current season data
                        combined_features.loc[i, 'current_form_factor'] = 0
        
        # Fill missing values
        combined_features['current_form_factor'] = combined_features['current_form_factor'].fillna(0)
        
        # Update the overall prediction score to include time-based factors
        combined_features['prediction_score'] = (
            combined_features['saudi_performance_score'] * 0.4 +
            combined_features['driver_constructor_synergy'] * 0.4 +
            combined_features['experience_factor'] * 0.1 +
            combined_features['saudi_recency_factor'] * 0.05 +
            combined_features['current_form_factor'] * 0.05
        )
        
        # Save combined features
        combined_features.to_csv(os.path.join(self.output_dir, 'combined_features.csv'), index=False)
        
        return combined_features
    
    def create_training_data(self):
        """Create training data for the model.
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        print("Creating training data...")
        
        # Load race results for training
        race_results = self._load_csv_file('race_results.csv')
        
        if race_results is None:
            print("No race results data found!")
            return None, None
        
        # Filter for Saudi Arabian GP results
        saudi_results = race_results.copy()
        
        # Create features for training data
        features = []
        
        for _, row in saudi_results.iterrows():
            driver_id = row['driver_id']
            constructor_id = row['constructor_id']
            year = row['year']
            
            # Get results from previous years for this driver
            prev_results = saudi_results[
                (saudi_results['driver_id'] == driver_id) & 
                (saudi_results['year'] < year)
            ]
            
            # Calculate features
            prev_saudi_races = len(prev_results)
            prev_saudi_best = prev_results['finish_position'].min() if not prev_results.empty else None
            prev_saudi_avg = prev_results['finish_position'].mean() if not prev_results.empty else None
            
            # Get grid position for this race
            grid_position = row['grid_position']
            
            # Create feature entry
            feature_entry = {
                'driver_id': driver_id,
                'constructor_id': constructor_id,
                'year': year,
                'grid_position': grid_position,
                'prev_saudi_races': prev_saudi_races,
                'prev_saudi_best': prev_saudi_best,
                'prev_saudi_avg': prev_saudi_avg,
                'finish_position': row['finish_position']  # Target variable
            }
            
            features.append(feature_entry)
        
        # Convert to DataFrame
        training_df = pd.DataFrame(features)
        
        # Fill missing values
        training_df['prev_saudi_races'] = training_df['prev_saudi_races'].fillna(0)
        training_df['prev_saudi_best'] = training_df['prev_saudi_best'].fillna(20)  # Default to back of grid
        training_df['prev_saudi_avg'] = training_df['prev_saudi_avg'].fillna(20)  # Default to back of grid
        
        # Save training data
        training_df.to_csv(os.path.join(self.output_dir, 'training_data.csv'), index=False)
        
        # Split into features and target
        X = training_df.drop('finish_position', axis=1)
        y = training_df['finish_position']
        
        return X, y
    
    def engineer_all_features(self):
        """Engineer all features for the prediction model."""
        print("Starting feature engineering...")
        
        # Engineer driver features
        driver_features = self.engineer_driver_features()
        
        # Engineer constructor features
        constructor_features = self.engineer_constructor_features()
        
        # Engineer combined features
        combined_features = self.engineer_combined_features()
        
        # Create training data
        X, y = self.create_training_data()
        
        print("Feature engineering completed!")
        
        return {
            'driver_features': driver_features,
            'constructor_features': constructor_features,
            'combined_features': combined_features,
            'X': X,
            'y': y
        }


if __name__ == "__main__":
    engineer = F1FeatureEngineer()
    features = engineer.engineer_all_features()
