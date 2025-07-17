# Total Sales Prediction for Cinema Tickets

## Overview
This project focuses on predicting total sales for cinema tickets using a dataset that includes features such as tickets sold, occupancy percentage, capacity, show time, ticket price, and ticket use. The goal is to explore the relationships between these features and total sales, identify the most influential factors, and build predictive models to forecast sales accurately. Two models were implemented and compared: a Linear Regression model and a k-Nearest Neighbors (KNN) Regressor. The KNN model outperformed the Linear Regression model, indicating that non-linear relationships in the data were better captured by the KNN approach.

## Project Structure
| File | Purpose |
|------|---------|
| `total-sales-prediction-for-cinema-tickets.ipynb` | The main Jupyter Notebook containing data loading, preprocessing, exploratory data analysis (EDA), visualization, and machine learning model implementation. |
| `cinemaTicket_Ref.csv` | The dataset used for analysis, containing features like tickets sold, occupancy percentage, capacity, show time, ticket price, ticket use, and total sales. |

## Dependencies
To run the project, ensure the following Python libraries are installed:
- `numpy`: For numerical operations and array manipulations.
- `pandas`: For data manipulation and analysis.
- `seaborn`: For statistical data visualization (e.g., heatmaps, scatter plots).
- `matplotlib`: For creating visualizations.
- `scikit-learn`: For machine learning model implementation and evaluation.

Install these libraries using:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Data Source
The dataset is sourced from Kaggle (dataset ID: 108707) and is available as `cinemaTicket_Ref.csv`. It includes features such as:
- `tickets_sold`
- `occu_perc` (occupancy percentage)
- `capacity`
- `show_time`
- `ticket_price`
- `ticket_use`
- `total_sales` (target variable)

### Preprocessing Steps
- **Loading**: The dataset was loaded using `pandas.read_csv`.
- **Missing Values**: Missing values in `occu_perc` and `capacity` were filled with their respective means.
- **Normalization**: The target variable (`total_sales`) was normalized using a logarithmic scale to handle skewness.
- **Scaling**: Features were scaled to a range of (-1, 1) using `MinMaxScaler`.

## Methodology
The project follows a structured approach to predict total sales:

1. **Data Preprocessing**:
   - Loaded the dataset and checked for missing values, filling them with column means.
   - Normalized the target variable (`total_sales`) using `np.log10` to address skewness.
   - Generated descriptive statistics using `df.describe()`.

2. **Exploratory Data Analysis (EDA)**:
   - Created a correlation heatmap using `seaborn.heatmap` to identify features strongly correlated with `total_sales`. Key correlations include:
     - `tickets_sold`: 0.92
     - `show_time`: 0.51
     - `capacity`: 0.38
   - Visualized relationships using scatter plots for highly correlated features.

3. **Feature Selection**:
   - Selected features with high correlation to `total_sales`: `tickets_sold`, `occu_perc`, `capacity`, `show_time`, `ticket_price`, and `ticket_use`.

4. **Data Splitting**:
   - Split the dataset into training (70%) and testing (30%) sets using `train_test_split`.

5. **Data Scaling**:
   - Applied `MinMaxScaler` to scale features to a range of (-1, 1) for consistent model input.

6. **Model Building**:
   - **Linear Regression**: Implemented using `sklearn.linear_model.LinearRegression` with selected features.
   - **K-Nearest Neighbors (KNN) Regressor**: Implemented using `sklearn.neighbors.KNeighborsRegressor` with 3 neighbors and two weighting schemes ("uniform" and "distance").

7. **Model Evaluation**:
   - Evaluated models using:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R-squared (R²)
   - Visualized actual vs. predicted values using distribution plots (`seaborn.distplot`).

## Results
- **Correlation Analysis**: Features like `tickets_sold`, `show_time`, and `capacity` showed strong correlations with `total_sales`.
- **Model Performance**:
  - The Linear Regression model provided reasonable predictions but was outperformed by the KNN Regressor.
  - The KNN Regressor achieved lower MAE, MSE, and RMSE, and a higher R², indicating better predictive performance.
  - Detailed metrics are available in the notebook.
- **Conclusion**: The KNN Regressor better captured non-linear relationships in the data, suggesting that the underlying patterns in cinema ticket sales are complex.

## How to Run

### Locally
1. Install Python 3.7.12 from [python.org](https://www.python.org/downloads/release/python-3712/).
2. Install the required libraries:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn jupyter
   ```
3. Download the dataset `cinemaTicket_Ref.csv` from [Kaggle dataset ID 108707](https://www.kaggle.com/datasets/108707).
4. Place the dataset in the same directory as the notebook or update the file path in the `pd.read_csv()` function.
5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
6. Open `total-sales-prediction-for-cinema-tickets.ipynb` and run it cell by cell.

### Troubleshooting
- **FileNotFoundError**: Ensure `cinemaTicket_Ref.csv` is in the correct directory or update the file path.
- **Library Issues**: Verify all dependencies are installed.
- **Plot Display**: Ensure `%matplotlib inline` is included for local execution.

## Contact
For questions or contributions, please open an issue or submit a pull request on GitHub.
