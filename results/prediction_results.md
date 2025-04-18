# 2025 Saudi Arabian Grand Prix - Prediction Results

## Model Performance Metrics

| Model | MAE | RMSE | Exact Position Accuracy | Within 1 Position | Within 3 Positions |
|-------|-----|------|------------------------|-------------------|-------------------|
| Random Forest | 2.76 | 3.53 | 12.50% | 31.25% | 68.75% |
| XGBoost | 2.76 | 3.40 | 18.75% | 25.00% | 75.00% |
| Gradient Boosting | 2.80 | 3.42 | 12.50% | 31.25% | 68.75% |

**Best Model: XGBoost**

## Predicted Leaderboard

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
| 11 | Nico Hülkenberg | 1:32.154 | 11 |
| 12 | Pierre Gasly | 1:31.834 | 10 |
| 13 | Lance Stroll | 1:32.192 | 13 |
| 14 | Logan Sargeant | 1:32.085 | 23 |
| 15 | Esteban Ocon | 1:32.065 | 14 |
| 16 | Valtteri Bottas | 1:32.332 | 22 |
| 17 | Oliver Bearman | 1:32.424 | 18 |
| 18 | Alexander Albon | 1:32.210 | 16 |
| 19 | Jack Doohan | 1:32.111 | 24 |
| 20 | Liam Lawson | 1:32.442 | 21 |
| 21 | Yuki Tsunoda | 1:32.123 | 12 |
| 22 | Daniel Ricciardo | 1:32.230 | 17 |
| 23 | Guanyu Zhou | 1:32.314 | 20 |
| 24 | Franco Colapinto | 1:32.221 | 19 |

## Key Insights

1. **Qualifying vs Race Performance**:
   - Lando Norris qualified 2nd but finished 4th
   - Carlos Sainz dropped from 5th on the grid to 9th
   - Kevin Magnussen improved dramatically from 15th to 8th

2. **Time-Based Feature Impact**:
   - Drivers with recent strong performances at Jeddah showed better predictions
   - Current season form was a significant factor in the final predictions

3. **Lap Time Analysis**:
   - Top 3 finishers all achieved lap times under 1:31.000
   - Gap between 1st and 2nd (0.237 seconds) is realistic for the Jeddah circuit
   - Midfield (positions 10-15) is very competitive with small time differences

## Model Execution Details

- **Execution Time**: 70.65 seconds
- **Date of Prediction**: April 18, 2025
- **Data Sources**: Ergast API (2018-2024 seasons)
