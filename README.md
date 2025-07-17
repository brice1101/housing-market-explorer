# UK Housing Market Analysis & Forecasting

This project explores and models trends in UK housing prices using real-world data from the HM Land Registry. It involves exploratory data analysis (EDA), data cleaning, feature engineering, and predictive modeling to understand market dynamics and forecast average house prices.

## Project Structure

```bash

.
|-- README.md
|-- data
|   |-- processed.csv.gz
|   `-- raw
|       `-- uk-hpi.csv.gz
|-- models
|-- notebooks
|   |-- EDA.ipynb
|   |-- model_comparison.ipynb
|   `-- rf_tuning.ipynb
|-- requirements.txt
`-- src
    |-- pipeline_lasso.py
    |-- pipeline_linear.py
    |-- pipeline_random_forest.py
    |-- pipeline_ridge.py
    |-- pipelines
    |   |-- __pycache__
    |   |   `-- base_pipeline.cpython-313.pyc
    |   `-- base_pipeline.py
    `-- utils
        |-- __pycache__
        |   `-- data_utils.cpython-313.pyc
        `-- data_utils.py

```

## Models

Trained models are **not included in this repository** because of their large size. The ```/models/``` directory is excluded in ```.gitignore```.

### To Train a Model

You can train and save a model by running the appropriate pipeline script:

```bash
python src/pipeline_linear.py
python src/pipeline_random_forest.py
```

Each script will save the trained model to:

```bash
/models/<model_name>.pkl
```

You can then use the model in analyses or notebooks.

## Features

- **Data cleaning:** Handles missing values, regional aggregation, and removes redundancy.
- **Feature engineering:**
    - Percent change over time
    - Categorical Encoding
- **Exploratory Data Analysis**: Heatmaps, price trends by region, correlation studies
- **Modeling:**
    - Linear regression baseline
    - Forecasting price trends
- **Compression:** Uses ```.csv.gz``` to keep versioned files GitHub-friendly

## Key Insights

## Using the repo

**1. Clone the repo**

```bash
git clone https://github.com/brice1101/housing-market-explorer.git
cd housing-market-explorer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**

## Data

Raw data sourced from:
- [HM Land Registry Open Data](https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-january-2024)
- Regional housing indices by dwelling type
- Sales volume and transaction type breakdowns

**Note**: Due to GitHub size limits, raw ```.csv``` files are not committed. Compressed, cleaned files are available under ```/data```.

## Next Steps

- Revisit feature engineering
- Try XGBoost and LightBGM
- Try Cross-validation instead of year split
- Try ensembling (e.g., Ridge + Random Forest)
- Implement time-series models (ARIMA, Prophet)
- Visualize forecasts on regional maps
- Add interactive dashboards with Plotly/Dash or Streamlit

## Author
**Brandon Lodge** <br/>
Computer Science with Artificial Intelligence Graduate | [LinkedIn](https://www.linkedin.com/in/brandon-lodge-6361401b8/) <br/>
Open to work in Data Analysis, Machine Learning, or Research

## License

This project is open-source under the MIT license.

