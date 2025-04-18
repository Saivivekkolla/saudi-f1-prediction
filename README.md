# Saudi Arabian Grand Prix F1 Leaderboard Prediction

This repository contains a machine learning model that predicts the finishing order (leaderboard) for the Saudi Arabian Grand Prix F1 race to be held from 18 to 20 April 2025.

## Project Overview
The model analyzes historical F1 data including:
- Previous race results at the Jeddah Corniche Circuit
- Driver performance statistics
- Team performance metrics
- Qualifying results
- Circuit characteristics
- Weather conditions (if available)

## Repository Structure
```
saudi-f1-prediction/
├── data/                # Data files
│   ├── raw/             # Raw data from APIs
│   └── processed/       # Processed datasets
├── models/              # Trained model files
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/            # Data collection and processing scripts
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Model training scripts
│   └── visualization/   # Visualization scripts
├── .gitignore           # Git ignore file
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup file
└── README.md            # Project documentation
```

## Setup Instructions

### Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
# Run the prediction script
python src/models/predict_leaderboard.py
```

## Data Sources
- [Ergast Motor Racing Developer API](http://ergast.com/mrd/)
- [Formula 1 Official Website](https://www.formula1.com/)
- Historical race data from previous Saudi Arabian Grand Prix events

## Model Performance
The model's accuracy is evaluated using:
- Mean Absolute Error (MAE) for position predictions
- Accuracy of podium predictions (top 3 finishers)

## License
MIT License

## Author
Created on April 18, 2025
