"""
Data collection module for F1 race prediction.
This script fetches data from the Ergast API and other sources.
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

class F1DataCollector:
    """Class to collect F1 data from various sources."""
    
    def __init__(self, data_dir='../../data/raw'):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to store raw data files
        """
        self.data_dir = data_dir
        self.base_url = 'http://ergast.com/api/f1'
        self.headers = {'User-Agent': 'F1-Prediction-Model/1.0'}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the Ergast API.
        
        Args:
            endpoint: API endpoint to request
            params: Query parameters
            
        Returns:
            JSON response data
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None
    
    def get_seasons(self, start_year=2018, end_year=2024):
        """Get F1 seasons data.
        
        Args:
            start_year: First season year to fetch
            end_year: Last season year to fetch
            
        Returns:
            DataFrame with seasons data
        """
        seasons_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching season data for {year}...")
            data = self._make_request(f"{year}.json")
            
            if data and 'MRData' in data:
                season_info = data['MRData']['RaceTable']
                seasons_data.append(season_info)
            
            # Be nice to the API
            time.sleep(0.5)
        
        # Save raw data
        with open(os.path.join(self.data_dir, 'seasons.json'), 'w') as f:
            json.dump(seasons_data, f)
            
        return seasons_data
    
    def get_race_results(self, circuit_id='jeddah', start_year=2021, end_year=2024):
        """Get race results for a specific circuit.
        
        Args:
            circuit_id: Circuit identifier (default: jeddah for Saudi Arabian GP)
            start_year: First year to fetch
            end_year: Last year to fetch
            
        Returns:
            DataFrame with race results
        """
        all_results = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching {circuit_id} race results for {year}...")
            
            # Try to get results by circuit ID first
            data = self._make_request(f"{year}/circuits/{circuit_id}/results.json")
            
            # If no results, try by country name (Saudi Arabia)
            if not data or 'MRData' not in data or int(data['MRData']['total']) == 0:
                data = self._make_request(f"{year}/saudiarabia/results.json")
            
            if data and 'MRData' in data and int(data['MRData']['total']) > 0:
                race_data = data['MRData']['RaceTable']['Races']
                if race_data:
                    all_results.append(race_data[0])
            
            # Be nice to the API
            time.sleep(0.5)
        
        # Save raw data
        with open(os.path.join(self.data_dir, f'{circuit_id}_results.json'), 'w') as f:
            json.dump(all_results, f)
        
        return all_results
    
    def get_qualifying_results(self, circuit_id='jeddah', start_year=2021, end_year=2024):
        """Get qualifying results for a specific circuit.
        
        Args:
            circuit_id: Circuit identifier (default: jeddah for Saudi Arabian GP)
            start_year: First year to fetch
            end_year: Last year to fetch
            
        Returns:
            DataFrame with qualifying results
        """
        all_qualifying = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching {circuit_id} qualifying results for {year}...")
            
            # Try to get results by circuit ID first
            data = self._make_request(f"{year}/circuits/{circuit_id}/qualifying.json")
            
            # If no results, try by country name (Saudi Arabia)
            if not data or 'MRData' not in data or int(data['MRData']['total']) == 0:
                data = self._make_request(f"{year}/saudiarabia/qualifying.json")
            
            if data and 'MRData' in data and int(data['MRData']['total']) > 0:
                qualifying_data = data['MRData']['RaceTable']['Races']
                if qualifying_data:
                    all_qualifying.append(qualifying_data[0])
            
            # Be nice to the API
            time.sleep(0.5)
        
        # Save raw data
        with open(os.path.join(self.data_dir, f'{circuit_id}_qualifying.json'), 'w') as f:
            json.dump(all_qualifying, f)
        
        return all_qualifying
    
    def get_driver_standings(self, year=2024):
        """Get current driver standings.
        
        Args:
            year: Year to fetch standings for
            
        Returns:
            DataFrame with driver standings
        """
        print(f"Fetching driver standings for {year}...")
        data = self._make_request(f"{year}/driverStandings.json")
        
        if data and 'MRData' in data:
            standings_data = data['MRData']['StandingsTable']['StandingsLists']
            
            # Save raw data
            with open(os.path.join(self.data_dir, f'driver_standings_{year}.json'), 'w') as f:
                json.dump(standings_data, f)
            
            return standings_data
        
        return None
    
    def get_constructor_standings(self, year=2024):
        """Get current constructor standings.
        
        Args:
            year: Year to fetch standings for
            
        Returns:
            DataFrame with constructor standings
        """
        print(f"Fetching constructor standings for {year}...")
        data = self._make_request(f"{year}/constructorStandings.json")
        
        if data and 'MRData' in data:
            standings_data = data['MRData']['StandingsTable']['StandingsLists']
            
            # Save raw data
            with open(os.path.join(self.data_dir, f'constructor_standings_{year}.json'), 'w') as f:
                json.dump(standings_data, f)
            
            return standings_data
        
        return None
    
    def get_drivers(self, year=2024):
        """Get all drivers for a season.
        
        Args:
            year: Year to fetch drivers for
            
        Returns:
            DataFrame with drivers
        """
        print(f"Fetching drivers for {year}...")
        data = self._make_request(f"{year}/drivers.json")
        
        if data and 'MRData' in data:
            drivers_data = data['MRData']['DriverTable']['Drivers']
            
            # Save raw data
            with open(os.path.join(self.data_dir, f'drivers_{year}.json'), 'w') as f:
                json.dump(drivers_data, f)
            
            return drivers_data
        
        return None
    
    def get_constructors(self, year=2024):
        """Get all constructors for a season.
        
        Args:
            year: Year to fetch constructors for
            
        Returns:
            DataFrame with constructors
        """
        print(f"Fetching constructors for {year}...")
        data = self._make_request(f"{year}/constructors.json")
        
        if data and 'MRData' in data:
            constructors_data = data['MRData']['ConstructorTable']['Constructors']
            
            # Save raw data
            with open(os.path.join(self.data_dir, f'constructors_{year}.json'), 'w') as f:
                json.dump(constructors_data, f)
            
            return constructors_data
        
        return None
    
    def get_circuit_info(self, circuit_id='jeddah'):
        """Get circuit information.
        
        Args:
            circuit_id: Circuit identifier
            
        Returns:
            Circuit information
        """
        print(f"Fetching circuit information for {circuit_id}...")
        data = self._make_request(f"circuits/{circuit_id}.json")
        
        if data and 'MRData' in data:
            circuit_data = data['MRData']['CircuitTable']['Circuits']
            
            # Save raw data
            with open(os.path.join(self.data_dir, f'circuit_{circuit_id}.json'), 'w') as f:
                json.dump(circuit_data, f)
            
            return circuit_data
        
        return None
    
    def collect_all_data(self):
        """Collect all necessary data for the prediction model."""
        print("Starting comprehensive data collection...")
        
        # Get seasons data
        self.get_seasons()
        
        # Get Saudi Arabian GP specific data
        self.get_race_results()
        self.get_qualifying_results()
        self.get_circuit_info()
        
        # Get current season data
        self.get_driver_standings()
        self.get_constructor_standings()
        self.get_drivers()
        self.get_constructors()
        
        print("Data collection completed!")


if __name__ == "__main__":
    collector = F1DataCollector()
    collector.collect_all_data()
