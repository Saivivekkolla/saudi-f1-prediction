# Saudi Arabian Grand Prix F1 Prediction Model

![F1 Prediction](https://img.shields.io/badge/F1-Prediction-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)

A comprehensive machine learning model that predicts the finishing order and lap times for the 2025 Saudi Arabian Grand Prix F1 race at the Jeddah Corniche Circuit.

## 📊 Project Overview

This project uses historical F1 data and advanced machine learning techniques to predict:
- Race finishing positions
- Estimated lap times
- Performance factors for each driver
- Starting grid positions impact

The model incorporates time-based features that give more weight to recent performances and current season form, providing a more accurate prediction of race outcomes.

## 🏗️ Project Structure

```
saudi-f1-prediction/
├── data/                # Data storage
│   ├── raw/             # Raw data from APIs
│   └── processed/       # Processed datasets
├── models/              # Trained model files and visualizations
├── src/                 # Source code
│   ├── data/            # Data collection & processing
│   │   ├── collect_data.py    # API data collection
│   │   └── process_data.py    # Data preprocessing
│   ├── features/        # Feature engineering
│   │   └── build_features.py  # Feature creation
│   ├── models/          # Model training & prediction
│   │   ├── train_model.py     # Model training
│   │   └── predict_leaderboard.py  # Prediction generation
│   ├── visualization/   # Result visualization
│   │   └── visualize.py       # Visualization tools
│   └── main.py          # Main execution script
├── .gitignore           # Git ignore file
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup file
└── README.md            # Project documentation
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12 or higher
- Git
- Internet connection (for data collection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saivivekkolla/saudi-f1-prediction.git
   cd saudi-f1-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏎️ Running the Model

### Full Pipeline Execution

To run the complete prediction pipeline (data collection, processing, feature engineering, model training, and prediction):

```bash
python src/main.py
```

This will:
1. Collect F1 data from the Ergast API
2. Process the raw data
3. Engineer features for prediction
4. Train and evaluate multiple models (Random Forest, XGBoost, Gradient Boosting)
5. Generate predictions for the 2025 Saudi Arabian Grand Prix
6. Create visualizations
7. Save results to the `data/processed/` and `models/` directories

### Output

The model produces:
- Terminal output with the predicted leaderboard
- CSV file with detailed predictions (`data/processed/saudi_gp_leaderboard.csv`)
- Visualization images in the `models/` directory:
  - `saudi_gp_leaderboard.png`: Predicted finishing positions
  - `saudi_gp_lap_times.png`: Predicted lap times
  - `saudi_gp_grid_vs_finish.png`: Starting grid vs. finishing position
  - `driver_performance.png`: Driver performance metrics
  - `constructor_performance.png`: Constructor performance metrics
  - `prediction_factors.png`: Feature importance visualization
  - `time_based_features.png`: Impact of time-based features

## 🔍 Key Features

### Data Collection
- Historical race results from 2018-2024
- Saudi Arabian GP specific data
- Driver and constructor standings
- Qualifying results

### Feature Engineering
1. **Driver-specific Features**
   - Saudi GP performance history
   - Current season standings
   - Total races completed

2. **Constructor-specific Features**
   - Team performance metrics
   - Historical circuit performance

3. **Time-based Features**
   - `saudi_recency_factor`: Measures how recently a driver has raced at Saudi GP
   - `current_form_factor`: Captures driver's momentum in current season
   - Exponential decay calculation for experience relevance

### Prediction Models
- Random Forest Regressor
- XGBoost Regressor (typically best performer)
- Gradient Boosting Regressor

### Model Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Position Accuracy (Exact, Within 1, Within 3 positions)

## 📊 Model Performance

Current model performance:
- Exact position accuracy: ~18.75%
- Within 1 position accuracy: ~25.00%
- Within 3 positions accuracy: ~75.00%

The XGBoost model typically performs best with:
- MAE: ~2.76
- RMSE: ~3.40

## 🏁 Prediction Results

### 2025 Saudi Arabian Grand Prix - Predicted Leaderboard

| Position | Driver | Lap Time | Starting Grid |
|----------|--------|----------|---------------|
| 1 | Max Verstappen | 1:30.423 | 1 |
| 2 | Charles Leclerc | 1:30.660 | 3 |
| 3 | Oscar Piastri | 1:30.800 | 4 |
| 4 | Lando Norris | 1:31.322 | 2 |
| 5 | Lewis Hamilton | 1:31.635 | 7 |
| 6 | George Russell | 1:31.675 | 6 |
| 7 | Sergio Pérez | 1:31.646 | 8 |
| 8 | Kevin Magnussen | 1:32.248 | 15 |
| 9 | Carlos Sainz | 1:31.928 | 5 |
| 10 | Fernando Alonso | 1:32.278 | 9 |

[View complete results](results/prediction_results.md)

### Visualizations

#### Predicted Leaderboard
![Predicted Leaderboard](results/images/saudi_gp_leaderboard.png)

#### Predicted Lap Times
![Predicted Lap Times](results/images/saudi_gp_lap_times.png)

#### Grid vs Finishing Position
![Grid vs Finish](results/images/saudi_gp_grid_vs_finish.png)

#### Driver Performance Factors
![Driver Performance](results/images/driver_performance.png)

#### Time-based Features Impact
![Time-based Features](results/images/time_based_features.png)

## 📚 Data Sources

- [Ergast Motor Racing Developer API](http://ergast.com/mrd/): Historical F1 race data
- [Formula 1 Official Website](https://www.formula1.com/): Supplementary information
- Historical race data from previous Saudi Arabian Grand Prix events (2021-2024)

## 🔮 Future Improvements

- Incorporate more granular weather data
- Add more sophisticated time decay calculations
- Implement ensemble prediction techniques
- Create more advanced feature engineering
- Add driver-specific lap time modeling

## 📄 License

MIT License

## 👨‍💻 Author

Created by KOLLA SAI VIVEK on April 18, 2025

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
