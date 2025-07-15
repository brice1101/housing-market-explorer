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
|   `-- linear_model.pkl
|-- notebooks
|   |-- EDA.ipynb
|   `-- modeling.ipynb
|-- requirements.txt
`-- src
    `-- pipeline.py
```

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

- Fine-tune existing model
- Try non-linear ensemble models and compare with linreg
- Implement time-series models (ARIMA, Prophet)
- Visualize forecasts on regional maps
- Add interactive dashboards with Plotly/Dash or Streamlit

## Author
**Brandon Lodge** <br/>
Computer Science with Artificial Intelligence Graduate | [LinkedIn](https://www.linkedin.com/in/brandon-lodge-6361401b8/) <br/>
Open to work in Data Analysis, Machine Learning, or Research

## License

This project is open-source under the MIT license.

