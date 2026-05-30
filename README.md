# Multi-Agent Dynamic Tariff Optimization for EV Charging Networks

This project implements an autonomous multi-agent system that dynamically optimizes electricity tariffs across EV charging networks to maximize operator revenue while reducing peak congestion.

## System Architecture

The system uses three cooperating AI agents:
1. **Demand Prediction**: Forecasts hourly utilization and energy demand load using a Random Forest model.
2. **Tariff Pricing**: Employs a constrained mathematical optimizer to set dynamic tariffs (₹9–₹30/kWh).
3. **Monitoring & Learning**: Evaluates pricing decisions and provides feedback over multiple simulation episodes.

## Datasets
- **ACN-Data**: Charging sessions from Caltech & JPL campuses.
- **UrbanEV**: Spatial-temporal charging demand from Shenzhen, China.
*(Aggregated into a unified hourly schema featuring 192k records across 301 stations)*

## Key Results
- **Revenue Gain**: +23.5% compared to fixed pricing baselines.
- **Demand Prediction**: Achieved an R² score of 0.9999.
- **Utilization**: Improved average charging infrastructure utilization by +2.8%.
- **Off-Peak Uplift**: +8.7% estimated increase in sessions during off-peak hours.

## Setup & Execution

### Prerequisites & Installation
- Python 3.9+ required

```bash
# Clone the repository
git clone https://github.com/ayush17me/EV_dynamic_tariff_optimization.git
cd EV_dynamic_tariff_optimization

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows (use `source .venv/bin/activate` for Mac/Linux)

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

The core project logic is contained in Jupyter notebooks. Launch a Jupyter session:

```bash
jupyter notebook
```

Execute the notebooks in the following sequential order from the `notebooks/` directory:
1. `01_data_preprocessing.ipynb`
2. `02_eda.ipynb`
3. `03_demand_prediction_agent.ipynb`
4. `04_tariff_pricing_agent.ipynb`
5. `05_monitoring_agent.ipynb`
