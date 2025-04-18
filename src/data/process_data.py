"""
Data preprocessing module for F1 race prediction.
This script processes raw data into features for the machine learning model.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class F1DataProcessor:
    """Class to process F1 data for the prediction model."""
    
    def __init__(self, raw_data_dir='../../data/raw', processed_data_dir='../../data/processed'):
        """Initialize the data processor.
        
        Args:
            raw_data_dir: Directory with raw data files
            processed_data_dir: Directory to store processed data files
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Create processed data directory if it doesn't exist
        os.makedirs(processed_data_dir, exist_ok=True)
    
    def _load_json_file(self, filename):
        """Load data from a JSON file.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            Loaded data or None if file doesn't exist
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def process_race_results(self):
        """Process race results data.
        
        Returns:
            DataFrame with processed race results
        """
        print("Processing race results...")
        
        # Load raw race results
        results_data = self._load_json_file('jeddah_results.json')
        if not results_data:
            print("No race results data found!")
            return None
        
        # Process race results
        all_results = []
        
        for race in results_data:
            year = race['season']
            race_date = race['date']
            
            for result in race['Results']:
                driver_id = result['Driver']['driverId']
                driver_name = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                constructor_id = result['Constructor']['constructorId']
                
                # Extract position data
                grid_pos = int(result['grid']) if result['grid'].isdigit() else None
                finish_pos = int(result['position']) if result['position'].isdigit() else None
                
                # Calculate position change
                pos_change = grid_pos - finish_pos if grid_pos and finish_pos else None
                
                # Extract timing data if available
                fastest_lap_rank = None
                fastest_lap_time = None
                if 'FastestLap' in result:
                    fastest_lap_rank = int(result['FastestLap']['rank'])
                    if 'Time' in result['FastestLap']:
                        fastest_lap_time = result['FastestLap']['Time']['time']
                
                # Extract status (finished, retired, etc.)
                status = result['status']
                
                # Create result entry
                result_entry = {
                    'year': year,
                    'race_date': race_date,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor_id': constructor_id,
                    'grid_position': grid_pos,
                    'finish_position': finish_pos,
                    'position_change': pos_change,
                    'fastest_lap_rank': fastest_lap_rank,
                    'fastest_lap_time': fastest_lap_time,
                    'status': status
                }
                
                all_results.append(result_entry)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save processed data
        results_df.to_csv(os.path.join(self.processed_data_dir, 'race_results.csv'), index=False)
        
        return results_df
    
    def process_qualifying_results(self):
        """Process qualifying results data.
        
        Returns:
            DataFrame with processed qualifying results
        """
        print("Processing qualifying results...")
        
        # Load raw qualifying results
        qualifying_data = self._load_json_file('jeddah_qualifying.json')
        if not qualifying_data:
            print("No qualifying data found!")
            return None
        
        # Process qualifying results
        all_qualifying = []
        
        for quali in qualifying_data:
            year = quali['season']
            quali_date = quali['date']
            
            for result in quali['QualifyingResults']:
                driver_id = result['Driver']['driverId']
                driver_name = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                constructor_id = result['Constructor']['constructorId']
                
                # Extract position
                position = int(result['position']) if result['position'].isdigit() else None
                
                # Extract Q1, Q2, Q3 times if available
                q1_time = result.get('Q1', None)
                q2_time = result.get('Q2', None)
                q3_time = result.get('Q3', None)
                
                # Create qualifying entry
                quali_entry = {
                    'year': year,
                    'quali_date': quali_date,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor_id': constructor_id,
                    'position': position,
                    'q1_time': q1_time,
                    'q2_time': q2_time,
                    'q3_time': q3_time
                }
                
                all_qualifying.append(quali_entry)
        
        # Convert to DataFrame
        qualifying_df = pd.DataFrame(all_qualifying)
        
        # Save processed data
        qualifying_df.to_csv(os.path.join(self.processed_data_dir, 'qualifying_results.csv'), index=False)
        
        return qualifying_df
    
    def process_driver_standings(self, year=2024):
        """Process driver standings data.
        
        Args:
            year: Year to process standings for
            
        Returns:
            DataFrame with processed driver standings
        """
        print(f"Processing driver standings for {year}...")
        
        # Load raw driver standings
        standings_data = self._load_json_file(f'driver_standings_{year}.json')
        if not standings_data:
            print(f"No driver standings data found for {year}!")
            return None
        
        # Process driver standings
        all_standings = []
        
        for standings_list in standings_data:
            season = standings_list['season']
            round_num = standings_list['round']
            
            for standing in standings_list['DriverStandings']:
                driver_id = standing['Driver']['driverId']
                driver_name = f"{standing['Driver']['givenName']} {standing['Driver']['familyName']}"
                
                position = int(standing['position'])
                points = float(standing['points'])
                wins = int(standing['wins'])
                
                # Get constructor info
                constructors = [c['constructorId'] for c in standing['Constructors']]
                constructor_id = constructors[0] if constructors else None
                
                # Create standing entry
                standing_entry = {
                    'season': season,
                    'round': round_num,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'constructor_id': constructor_id,
                    'position': position,
                    'points': points,
                    'wins': wins
                }
                
                all_standings.append(standing_entry)
        
        # Convert to DataFrame
        standings_df = pd.DataFrame(all_standings)
        
        # Save processed data
        standings_df.to_csv(os.path.join(self.processed_data_dir, f'driver_standings_{year}.csv'), index=False)
        
        return standings_df
    
    def process_constructor_standings(self, year=2024):
        """Process constructor standings data.
        
        Args:
            year: Year to process standings for
            
        Returns:
            DataFrame with processed constructor standings
        """
        print(f"Processing constructor standings for {year}...")
        
        # Load raw constructor standings
        standings_data = self._load_json_file(f'constructor_standings_{year}.json')
        if not standings_data:
            print(f"No constructor standings data found for {year}!")
            return None
        
        # Process constructor standings
        all_standings = []
        
        for standings_list in standings_data:
            season = standings_list['season']
            round_num = standings_list['round']
            
            for standing in standings_list['ConstructorStandings']:
                constructor_id = standing['Constructor']['constructorId']
                constructor_name = standing['Constructor']['name']
                
                position = int(standing['position'])
                points = float(standing['points'])
                wins = int(standing['wins'])
                
                # Create standing entry
                standing_entry = {
                    'season': season,
                    'round': round_num,
                    'constructor_id': constructor_id,
                    'constructor_name': constructor_name,
                    'position': position,
                    'points': points,
                    'wins': wins
                }
                
                all_standings.append(standing_entry)
        
        # Convert to DataFrame
        standings_df = pd.DataFrame(all_standings)
        
        # Save processed data
        standings_df.to_csv(os.path.join(self.processed_data_dir, f'constructor_standings_{year}.csv'), index=False)
        
        return standings_df
    
    def process_drivers(self, year=2024):
        """Process drivers data.
        
        Args:
            year: Year to process drivers for
            
        Returns:
            DataFrame with processed drivers
        """
        print(f"Processing drivers for {year}...")
        
        # Load raw drivers data
        drivers_data = self._load_json_file(f'drivers_{year}.json')
        if not drivers_data:
            print(f"No drivers data found for {year}!")
            return None
        
        # Process drivers data
        all_drivers = []
        
        for driver in drivers_data:
            driver_id = driver['driverId']
            driver_code = driver.get('code', '')
            driver_number = driver.get('permanentNumber', '')
            driver_name = f"{driver['givenName']} {driver['familyName']}"
            nationality = driver['nationality']
            
            # Create driver entry
            driver_entry = {
                'driver_id': driver_id,
                'driver_code': driver_code,
                'driver_number': driver_number,
                'driver_name': driver_name,
                'nationality': nationality
            }
            
            all_drivers.append(driver_entry)
        
        # Convert to DataFrame
        drivers_df = pd.DataFrame(all_drivers)
        
        # Save processed data
        drivers_df.to_csv(os.path.join(self.processed_data_dir, f'drivers_{year}.csv'), index=False)
        
        return drivers_df
    
    def process_constructors(self, year=2024):
        """Process constructors data.
        
        Args:
            year: Year to process constructors for
            
        Returns:
            DataFrame with processed constructors
        """
        print(f"Processing constructors for {year}...")
        
        # Load raw constructors data
        constructors_data = self._load_json_file(f'constructors_{year}.json')
        if not constructors_data:
            print(f"No constructors data found for {year}!")
            return None
        
        # Process constructors data
        all_constructors = []
        
        for constructor in constructors_data:
            constructor_id = constructor['constructorId']
            constructor_name = constructor['name']
            nationality = constructor['nationality']
            
            # Create constructor entry
            constructor_entry = {
                'constructor_id': constructor_id,
                'constructor_name': constructor_name,
                'nationality': nationality
            }
            
            all_constructors.append(constructor_entry)
        
        # Convert to DataFrame
        constructors_df = pd.DataFrame(all_constructors)
        
        # Save processed data
        constructors_df.to_csv(os.path.join(self.processed_data_dir, f'constructors_{year}.csv'), index=False)
        
        return constructors_df
    
    def process_all_data(self):
        """Process all collected data and prepare for feature engineering."""
        print("Starting data processing...")
        
        # Process race and qualifying data
        race_results = self.process_race_results()
        qualifying_results = self.process_qualifying_results()
        
        # Process current season data
        driver_standings = self.process_driver_standings()
        constructor_standings = self.process_constructor_standings()
        drivers = self.process_drivers()
        constructors = self.process_constructors()
        
        print("Data processing completed!")
        
        return {
            'race_results': race_results,
            'qualifying_results': qualifying_results,
            'driver_standings': driver_standings,
            'constructor_standings': constructor_standings,
            'drivers': drivers,
            'constructors': constructors
        }


if __name__ == "__main__":
    processor = F1DataProcessor()
    processed_data = processor.process_all_data()
