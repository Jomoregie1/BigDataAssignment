import calendar
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# Function to read the data from the CSV file
def read_data(filepath):
    """
        Read the data from a CSV file and return a pandas DataFrame.

        Args:
            filepath: path to the CSV file

        Returns:
            A pandas DataFrame containing the data from the CSV file.
        """
    data = pd.read_csv(filepath, skiprows=4)
    return data


# Function to rename the columns of the DataFrame
def rename_columns(data):
    """
        Rename the columns of a pandas DataFrame using a dictionary.

        Args:
            data: a pandas DataFrame to rename the columns of

        Returns:
            A pandas DataFrame with renamed columns.
        """

    # Dictionary to map old column names to new column names
    columns_names = {'BOROUGH': 'borough',
                     'NEIGHBORHOOD': 'neighborhood',
                     'BUILDING CLASS CATEGORY': 'building_class_category',
                     'TAX CLASS AT PRESENT': 'tax_class_present',
                     'BLOCK': 'block',
                     'LOT': 'lot',
                     'EASE-MENT': 'easement',
                     'BUILDING CLASS AT PRESENT': 'building_class_present',
                     'ADDRESS': 'address',
                     'APART\nMENT\nNUMBER': 'apartment_number',
                     'ZIP CODE': 'zip_code',
                     'RESIDENTIAL UNITS': 'residential_units',
                     'COMMERCIAL UNITS': 'commercial_units',
                     'TOTAL UNITS': 'total_units',
                     'LAND SQUARE FEET': 'land_square_feet',
                     'GROSS SQUARE FEET': 'gross_square_feet',
                     'YEAR BUILT': 'year_built',
                     'TAX CLASS AT TIME OF SALE': 'tax_class_time_sale',
                     'BUILDING CLASS AT TIME OF SALE': 'building_class_time_sale',
                     'SALE\nPRICE': 'sale_price',
                     'SALE DATE': 'sale_date'}
    # Rename columns using the dictionary above
    data = data.rename(columns=columns_names)
    return data


def remove_special_characters(data):
    """
    Remove special characters and convert sale_price column to integer.

    Args:
        data: a pandas DataFrame to clean the sale_price column

    Returns:
        A pandas DataFrame with cleaned sale_price column.
    """
    data['sale_price'] = data['sale_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(
        int)
    return data


def convert_sale_date(data):
    """
    Convert sale_date column to datetime format.

    Args:
        data: a pandas DataFrame to convert the sale_date column

    Returns:
        A pandas DataFrame with converted sale_date column.
    """
    data['sale_date'] = pd.to_datetime(data['sale_date'], format='%d/%m/%Y')
    return data


def replace_empty_values_with_NaN(data):
    """
    Replace empty values with NaN in select columns.

    Args:
        data: a pandas DataFrame to replace empty values

    Returns:
        A pandas DataFrame with replaced empty values.
    """
    data['neighborhood'] = data['neighborhood'].str.strip().replace('', np.nan)
    data['building_class_category'] = data['building_class_category'].str.strip().replace('', np.nan)
    data['tax_class_present'] = data['tax_class_present'].str.strip().replace('', np.nan)
    data['building_class_present'] = data['building_class_present'].str.strip().replace('', np.nan)
    data['building_class_time_sale'] = data['building_class_time_sale'].str.strip().replace('', np.nan)
    data['address'] = data['address'].str.strip().replace('', np.nan)
    return data


def replace_zero_with_NaN(data):
    """
    Replace all 0 values with NaN.

    Args:
        data: a pandas DataFrame to replace 0 values

    Returns:
        A pandas DataFrame with replaced 0 values.
    """
    data = data.replace(0, np.nan)
    return data


def drop_irrelevant_columns(data):
    """
    Drop irrelevant columns.

    Args:
        data: a pandas DataFrame to drop columns

    Returns:
        A pandas DataFrame with dropped columns.
    """
    data = data.drop(['borough', 'easement', 'apartment_number'], axis=1)
    return data


def remove_duplicates_and_missing_values(data):
    """
    Remove duplicate rows and rows with missing values.

    Args:
        data: a pandas DataFrame to remove duplicates and missing values

    Returns:
        A pandas DataFrame without duplicates and missing values.
    """
    data = data.drop_duplicates()
    data = data.dropna()
    return data


def remove_outliers(data):
    """
    Remove outliers using the interquartile range method.

    Args:
        data: a pandas DataFrame to remove outliers

    Returns:
        A pandas DataFrame without outliers.
    """
    q1 = data['sale_price'].quantile(0.25)
    q3 = data['sale_price'].quantile(0.75)
    iqr = q3 - q1
    data = data[~((data['sale_price'] < (q1 - 1.5 * iqr)) | (data['sale_price'] > (q3 + 1.5 * iqr)))]
    return data


def apply_logarithm(data):
    """
    Take the logarithm of the sale_price column.

    Args:
        data: a pandas DataFrame to apply logarithm on the sale_price column

    Returns:
        A pandas DataFrame with the logarithm applied to the sale_price column.
    """
    data['logprices'] = np.log(data['sale_price'] + 1)
    data.drop('sale_price', axis=1, inplace=True)
    return data


def normalize_numerical_columns(data, numerical_cols):
    """
    Normalize the numerical columns using the StandardScaler method.

    Args:
        data: a pandas DataFrame to normalize the numerical columns
        numerical_cols: a list of column names to be normalized

    Returns:
        A pandas DataFrame with normalized numerical columns.
    """

    # Create a MinMaxScaler object to scale numerical columns later
    scaler = MinMaxScaler()

    data[numerical_cols] = data[numerical_cols].replace(',', '', regex=True)
    data[numerical_cols] = data[numerical_cols].astype(float)
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data


def impute_missing_numerical_data(data):
    """
    Impute missing values in a pandas DataFrame.

    Args:
        data: a pandas DataFrame to impute missing values in

    Returns:
        A pandas DataFrame with imputed missing values.
    """

    # Impute missing numeric values with mean
    data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

    return data


def impute_missing_categorical_data(data):
    """
    Impute missing values in a pandas DataFrame.

    Args:
        data: a pandas DataFrame to impute missing values in

    Returns:
        A pandas DataFrame with imputed missing values.
    """

    categorical_columns = [
        "neighborhood",
        "building_class_category",
        "tax_class_present",
        "building_class_present",
        "building_class_time_sale",
        "address"
    ]

    # Impute missing categorical values with mode
    for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    return data


def preprocess_data_for_part_1(filepath):
    """
    Read, clean, and preprocess a CSV file containing real estate data for part one.

    Args:
        filepath: path to the CSV file

    Returns:
        A pandas DataFrame containing the cleaned and preprocessed data.
    """
    # Read the data from the CSV file
    data = read_data(filepath)
    print("********* MODEL-1  **************")
    print(f'This is the shape of the dataframe before cleaning:  {data.shape}')

    # Rename the columns of the DataFrame
    data = rename_columns(data)

    # Clean and preprocess data
    data = remove_special_characters(data)
    data = convert_sale_date(data)
    data = replace_empty_values_with_NaN(data)
    data = replace_zero_with_NaN(data)
    # Display missing data
    display_summary_of_missing_data(data)
    data = drop_irrelevant_columns(data)
    data = remove_duplicates_and_missing_values(data)
    data = remove_outliers(data)
    print(f'This is the shape of the dataframe after removing outliers:  {data.shape}')
    # To show the prices over time before log has been applied to the data.
    visualize_prices_across_time(data)
    data = apply_logarithm(data)

    # Define the numerical columns list
    numerical_cols = ['total_units', 'land_square_feet', 'gross_square_feet', 'logprices', 'year_built',
                      'tax_class_time_sale']

    # Normalize the numerical columns
    data = normalize_numerical_columns(data, numerical_cols)

    return data


def preprocess_data_for_part_2(filepath):
    """
        Read, clean, and preprocess a CSV file containing real estate data for part two.

        Args:
            filepath: path to the CSV file

        Returns:
            A pandas DataFrame containing the cleaned and preprocessed data.
        """

    # Define the numerical columns list
    numerical_cols = ['total_units', 'land_square_feet', 'gross_square_feet', 'logprices', 'year_built',
                      'tax_class_time_sale']

    # Read the data from the CSV file
    data = read_data(filepath)
    print("\n********* MODEL-2  **************")
    print(f'This is the shape of the data before cleaning:  {data.shape}')

    # Rename the columns of the DataFrame
    data = rename_columns(data)
    # Clean and preprocess data
    data = remove_special_characters(data)
    # Apply logarithm transformation to the sale_price column
    data = apply_logarithm(data)
    data = convert_sale_date(data)
    data = replace_empty_values_with_NaN(data)
    data = replace_zero_with_NaN(data)
    data = impute_missing_numerical_data(data)
    data = impute_missing_categorical_data(data)
    data = remove_duplicates_and_missing_values(data)
    data = normalize_numerical_columns(data, numerical_cols)
    data = drop_irrelevant_columns(data)
    data = dbscan_outlier_detection(data)
    print(f'This is the shape of the dataframe after removing outliers:  {data.shape}')

    return data


def visualize_prices_across_neighborhood(data):
    """
    Visualizes the distribution of prices of properties across neighborhoods using a box plot.

    Args:
        data (pandas.DataFrame): The dataset containing the sale prices and neighborhoods of properties.

    Returns:
        None.
    """
    plt.figure(figsize=(20, 10))
    # create a box plot
    sns.boxplot(x="neighborhood", y="logprices", data=data)
    plt.title("Price Distribution Across Neighborhoods")
    plt.xticks(rotation=90)
    plt.show()


def dbscan_outlier_detection(data, eps=0.5, min_samples=5):
    """
    Removes outliers from the dataset using the DBSCAN method.

    Args:
        data (pandas.DataFrame): The dataset to remove outliers from.
        eps (float): The maximum distance between two samples for them to be considered as in the same cluster.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        pandas.DataFrame: The dataset with outliers removed.
    """
    numerical_cols = data.select_dtypes(include=np.number).columns
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    preds = dbscan.fit_predict(data[numerical_cols])
    data_clean = data[preds != -1]

    return data_clean


def visualize_prices_across_time(data):
    """
    Visualizes the average sale prices of properties over time using a line plot.

    Args:
        data (pandas.DataFrame): The dataset containing the sale prices and sale dates of properties.

    Returns:
        None.
    """
    # group the data by year and month and calculate average sale price
    avg_prices = data.groupby([data['sale_date'].dt.year.rename('year'), data['sale_date'].dt.month.rename('month')])[
        'sale_price'].mean().reset_index()

    # convert numerical month values to month names
    avg_prices['month'] = avg_prices['month'].apply(lambda x: calendar.month_name[x])

    # specify the order of the months
    month_order = list(calendar.month_name)[1:]

    # convert month column to categorical data type with specified order
    avg_prices['month'] = pd.Categorical(avg_prices['month'], categories=month_order, ordered=True)

    # sort the data by year and month
    # avg_prices = avg_prices.sort_values(['year', 'month'])

    # plot the data using a line plot
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=avg_prices, x="month", y="sale_price", hue="year", marker="o")
    plt.title("Average Sale Prices Across Time")
    plt.xlabel("Month")
    plt.ylabel("Average Sale Price")
    plt.xticks(range(0, 12), calendar.month_name[1:13], rotation=45)

    # set the formatter for the y-axis to ScalarFormatter
    formatter = StrMethodFormatter('£{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Display the gridlines
    plt.grid(True)

    plt.gca().set_ylabel('Average sale price', labelpad=20)

    plt.show()


def visualize_scatter_matrix(data):
    """
    Plots a scatter matrix for specified columns of the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to plot.

    Returns:
    None.
    """

    sns.set_style('ticks')

    scatter_cols = ['total_units', 'land_square_feet', 'gross_square_feet', 'logprices']
    pd.plotting.scatter_matrix(data[scatter_cols], diagonal='hist', hist_kwds=None, grid=False)

    plt.show()


def visualise_regression_plot(data):
    """
       Visualize a regression line between log prices and each numerical column in the input data.
       Args:
           data (pandas.DataFrame): The input data to visualize.
       Returns:
           None.
       Raises:
           None.
       """
    # Select only the numerical columns in your data
    numerical_cols = data.select_dtypes(include=np.number).columns

    # Plot a regression line between log prices and each numerical column
    for col in numerical_cols:
        if col != 'logprices':  # Exclude logprices from the plot
            sns.regplot(x='logprices', y=col, data=data)
            plt.title(f"Regression plot against logprice and {col}")
            plt.show()


def visualize_corr_matrix(data):
    """
    Plots a correlation matrix heatmap for the numerical columns of the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to plot.

    Returns:
    None.
    """
    corr_matrix = data.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.show()


def visualize_boxplot(data):
    """
    Plots a boxplot for the numerical columns of the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to plot.

    Returns:
    None.
    """
    # Define the numerical columns list
    numerical_cols = ['total_units', 'land_square_feet', 'gross_square_feet', 'logprices', 'year_built',
                      'tax_class_time_sale']

    data[numerical_cols].boxplot()
    plt.title('Boxplot of Manhattan Real Estate Data')
    plt.show()


# TODO: what can you actually intrepret from this graph
def visualize_violinplot(data):
    """
    Plots a violin plot for the specified column of the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to plot.

    Returns:
    None.
    """
    sns.set(style="whitegrid")
    ax = sns.violinplot(x=data['logprices'])
    ax.set(xlabel='Total Units', title='Distribution of Total Units')
    plt.show()


def select_predictor_variables(data):
    """
    Selects the predictor variables from the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame containing the predictor variables.
    """
    x = data[['total_units', 'land_square_feet', 'gross_square_feet', 'year_built']]
    return sm.add_constant(x)


def fit_linear_model(x, y):
    """
    Fits a linear regression model to the input data.

    Parameters:
    X (pandas.DataFrame): The DataFrame containing the predictor variables.
    y (pandas.Series): The target variable.

    Returns:
    statsmodels.regression.linear_model.RegressionResultsWrapper: The results of the linear regression.
    """
    return sm.OLS(y, x).fit()


def split_data(x, y, test_size=0.3, random_state=42):
    """
    Splits the input data into training and testing sets.

    Parameters:
    X (pandas.DataFrame): The DataFrame containing the predictor variables.
    y (pandas.Series): The target variable.
    test_size (float): The fraction of the data to use for testing.
    random_state (int): The random seed to use.

    Returns:
    tuple: A tuple containing the training and testing sets for the predictor variables and target variable.
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_linear_regression_model(x_train, y_train):
    """
    Trains a linear regression model on the input data.

    Parameters:
    X_train (pandas.DataFrame): The training set for the predictor variables.
    y_train (pandas.Series): The training set for the target variable.

    Returns:
    sklearn.linear_model.LinearRegression: The trained linear regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    return lr_model


def predict_test_data(lr_model, x_test):
    """
    Predicts the target variable using the fitted linear regression model and test data.

    Args:
    - lr_model: the fitted linear regression model.
    - X_test: the test data used for prediction.

    Returns:
    - An array of predicted target values.
    """
    return lr_model.predict(x_test)


def calculate_residuals(y_test, y_pred):
    """
    Calculates the residuals between the true and predicted target values.

    Args:
    - y_test: the true target values.
    - y_pred: the predicted target values.

    Returns:
    - An array of residuals.
    """
    return y_test - y_pred


def calculate_cv_scores(lr_model, x, y, cv=5):
    """
    Calculates the cross-validation scores of a linear regression model.

    Args:
    - lr_model: the linear regression model to evaluate.
    - X: the feature data to use for evaluation.
    - y: the target variable to use for evaluation.
    - cv: the number of folds to use for cross-validation.

    Returns:
    - An array of cross-validation scores.
    """
    return cross_val_score(lr_model, x, y, cv=cv)


def plot_residuals_histogram(residuals):
    """
    Plots a histogram of residuals.

    Args:
    - residuals: an array of residuals.

    Returns:
    - None
    """
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()


def display_summary_of_missing_data(data):
    """
    Displays a bar plot of the count and percentage of missing values for each column in a DataFrame.

    Args:
    - data: the DataFrame to analyze.

    Returns:
    - None
    """
    sns.set(style="whitegrid")
    missing = data.isnull().sum()
    missing_perc = missing / len(data) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_perc})
    missing_df.sort_values(by='Count', ascending=False, inplace=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_df.index, y=missing_df['Count'], palette='Blues_d')
    plt.xticks(rotation=90)
    plt.xlabel('Columns')
    plt.ylabel('Number of missing values')
    plt.title('Missing values in the dataset')
    plt.show()


def select_features_with_rfe(data, n_features=5):
    """
    Selects the top `n_features` number of features with Recursive Feature Elimination (RFE).
    RFE is a feature selection method that recursively selects features by ranking them
    according to their importance and eliminating the least important ones. Linear Regression
    is used as the estimator to rank the features.

    Parameters:
        data (pd.DataFrame): The input data, must contain a numeric target column 'logprices'.
        n_features (int): The number of features to select.

    Returns:
        list: A list of the top `n_features` selected features based on RFE.
    """

    # Split data into x and y
    x = data.select_dtypes(include=np.number).drop('logprices', axis=1)
    y = data['logprices']

    # Define estimator (linear regression)
    estimator = LinearRegression()

    # Define RFE
    rfe = RFE(estimator, n_features_to_select=n_features)

    # Fit RFE to data
    rfe.fit(x, y)

    return x.columns[rfe.support_]


def random_forest_regression(x_train, y_train, x_test, y_test, n_estimators=100, max_depth=None):
    """
    Trains a Random Forest regression model using the training data (X_train and y_train).
    Returns the trained model and the mean squared error on the test set.
    """

    # Define the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train the model on the training data
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Calculate the mean squared error on the test set
    mse = mean_squared_error(y_test, y_pred)

    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()

    print("Mean CV Score:", mean_cv_score)

    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)

    return model, mse, y_pred


def evaluate_random_forest(n_estimators, max_depth):
    """
       Defines the objective function for Bayesian Optimization.
       Trains and evaluates a Random Forest model on the training data using the given hyperparameters.
       Returns the negative mean squared error on the cross-validation set (to be maximized by Bayesian Optimization).
       """
    model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    return cv_scores.mean()


def bayesian_Optimization(data, target_name, feature_names):
    """
    Trains and evaluates a random forest regression model using Bayesian Optimization.
    Args:
    data (pandas.  DataFrame): the data to use in the model.
    target_name (str): the name of the target variable.
    feature_names (list): a list of the names of the features to use in the model.
    Returns:
    The final trained model and the mean squared error on the test set.
    """

    # Select the features to use in the model
    x = data[feature_names]
    y = data[target_name]

    # Split the data into training and test sets
    x, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Define the search space for the hyperparameters
    pbounds = {'n_estimators': (10, 200),
               'max_depth': (2, 20)}

    # Run the Bayesian Optimization process
    optimizer = BayesianOptimization(
        f=evaluate_random_forest,
        pbounds=pbounds,
        random_state=42,
    )

    print("\nFine tuning hyper-parameters..............")
    # Perform the optimization
    optimizer.maximize(init_points=5, n_iter=20)

    # Print the best hyperparameters and the corresponding CV score
    best_params = optimizer.max['params']
    best_cv_score = optimizer.max['target']
    print("\nBest Hyperparameters:", best_params)
    print("Best CV Score:", best_cv_score)

    # Train the final model with the best hyperparameters
    final_model, final_mse, y_pred = random_forest_regression(x, y_train, x_test, y_test,
                                                              n_estimators=int(best_params['n_estimators']),
                                                              max_depth=int(best_params['max_depth']))

    mse = mean_squared_error(y_test, y_pred)
    regression_residuals_plot(y_test, y_pred)
    print("Final Model MSE:", mse)

    return final_model, mse


def regression_residuals_plot(y_test, y_pred):
    """
    Creates a scatter plot of the residuals vs. predicted values for a linear regression model.

    Parameters:
    y_test (array-like): True target variable values for the test data.
    y_pred (array-like): Predicted target variable values for the test data.

    Returns:
    None
    """
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()


def silhouetteScore(data):
    """
    Calculates the silhouette scores for different numbers of clusters in a KMeans clustering model.

    Parameters:
    data (pandas DataFrame): Data used to fit the KMeans clustering model.

    Returns:
    silhouette_scores (list): List of silhouette scores for each number of clusters.
    """
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data.select_dtypes(include=np.number))
        score = silhouette_score(data.select_dtypes(include=np.number), kmeans.labels_)
        silhouette_scores.append(score)

    print(f'A list of the silhouette scores for each number of clusters: {silhouette_scores}')
    return silhouette_scores


def visualize_silhouette_scores(scores):
    """
     Plots the silhouette scores for different numbers of clusters, ranging from 2 to 10.

     The silhouette score is a measure of how well each data point fits within its cluster, taking into
     account the average distance between the point and all other points within the same cluster, as well
     as the average distance between the point and all other points in the nearest neighboring cluster.
     A higher silhouette score indicates better cluster cohesion and separation.

     Args:
         scores (list): A list of silhouette scores, where the index (plus 2) corresponds to the number of clusters.

     Returns:
         None.
     """
    plt.plot(range(2, 11), scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.show()


def visualize_clusters(data, kmeans_labels, x_var, y_var, title, xlabel, ylabel):
    """
    Creates a scatter plot to visualize the clusters in a KMeans clustering model.

    Parameters:
    data (pandas DataFrame): Data used to fit the KMeans clustering model.
    kmeans_labels (array-like): Cluster labels generated by the KMeans clustering model.
    x_var (str): Column name for the x-axis variable.
    y_var (str): Column name for the y-axis variable.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.

    Returns:
    None
    """
    sns.scatterplot(x=data[x_var], y=data[y_var], hue=kmeans_labels, palette='Set1')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def create_cluster_dataframes(data, cluster_labels):
    """
    Creates a list of dataframes, where each dataframe contains the data for a single cluster.

    Parameters:
    data (pandas DataFrame): Data used to fit the KMeans clustering model.
    cluster_labels (array-like): Cluster labels generated by the KMeans clustering model.

    Returns:
    cluster_dataframes (list): List of dataframes, where each dataframe contains the data for a single cluster.
    """
    data_numeric = data.select_dtypes(include=np.number)
    data_numeric['cluster'] = cluster_labels

    cluster_dataframes = []
    num_clusters = len(set(cluster_labels))

    for i in range(num_clusters):
        cluster_data = data_numeric[data_numeric['cluster'] == i]
        cluster_dataframes.append(cluster_data)

    return cluster_dataframes


def split_and_train_clusters(cluster_dataframes, target='logprices'):
    """
      Split each cluster dataframe into training and testing sets, train a linear regression model on the training set,
      and store the trained model, training set, and testing set for each cluster in separate lists.

      Args:
          cluster_dataframes: List of dataframes, where each dataframe contains data for a single cluster.
          target: Name of the target variable. Default is 'logprices'.

      Returns:
          x_train_list: List of training sets for each cluster.
          x_test_list: List of testing sets for each cluster.
          y_train_list: List of target values for the training sets for each cluster.
          y_test_list: List of target values for the testing sets for each cluster.
          lr_models: List of trained linear regression models for each cluster.
      """
    x_train_list, x_test_list, y_train_list, y_test_list, lr_models = [], [], [], [], []

    for cluster_df in cluster_dataframes:
        x = cluster_df.drop([target, 'cluster'], axis=1)
        y = cluster_df[target]
        x_train, x_test, y_train, y_test = split_data(x, y)
        lr_model = train_linear_regression_model(x_train, y_train)

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        lr_models.append(lr_model)

    return x_train_list, x_test_list, y_train_list, y_test_list, lr_models


def evaluate_cluster_models(lr_models, x_test_list, y_test_list):
    """
    Evaluate the cluster-based linear regression models by computing the mean squared error (MSE) and mean absolute
    error (MAE) for each cluster model and returning the results as lists.

    Args: lr_models (list): A list of trained linear regression models, one for each cluster. X_test_list (list): A
    list of testing data for each cluster, with one array of predictor variables for each model. y_test_list (list):
    A list of testing data for each cluster, with one array of target variables for each model.

    Returns:
        tuple: A tuple containing two lists, the first containing the MSE for each cluster model and the second
        containing the MAE for each cluster model.
    """

    mse_list, mae_list = [], []

    for i, lr_model in enumerate(lr_models):
        y_pred = predict_test_data(lr_model, x_test_list[i])
        mse_list.append(mean_squared_error(y_test_list[i], y_pred))
        mae_list.append(mean_absolute_error(y_test_list[i], y_pred))

    return mse_list, mae_list


def compare_models_graphically(data, cluster_models, x_test_list, y_test_list):
    """
        Compare the performance of cluster-based linear regression models with an overall linear regression model
        using a scatter plot of true values against predicted values.

        Parameters:
        data (pd.DataFrame): The entire dataset, including the target variable.
        cluster_models (list): A list of trained linear regression models for each cluster.
        X_test_list (list): A list of test feature sets for each cluster.
        y_test_list (list): A list of test target variables for each cluster.

        Returns: None. Shows a scatter plot of true values against predicted values for the overall model and each
        cluster-based model.
    """

    # Split the entire dataset into training and testing sets
    x_train = data.select_dtypes(include=np.number).drop('logprices', axis=1)
    y = data['logprices']
    x_train, x_test, y_train, y_test = split_data(x_train, y)

    # Train the overall linear regression model
    lr_model = train_linear_regression_model(x_train, y_train)

    # Predict the test data
    y_pred = predict_test_data(lr_model, x_test)

    # Plot the overall model
    plt.scatter(y_test, y_pred, alpha=0.3, label="Overall Model")

    # Plot the cluster-based models
    for i in range(len(cluster_models)):
        y_pred_cluster = predict_test_data(cluster_models[i], x_test_list[i])
        plt.scatter(y_test_list[i], y_pred_cluster, alpha=0.3, label=f"Cluster {i}")

    # Add diagonal line
    min_value = min(data['logprices'])
    max_value = max(data['logprices'])
    plt.plot([min_value, max_value], [min_value, max_value], 'k--', lw=2, label="Ideal Fit")

    # Customize the plot
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs. Predictions')
    plt.legend(loc='upper left')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Read the data and preprocess it
    path = 'Manhattan12.csv'
    manhattan_data = preprocess_data_for_part_1(path)

    ''' PART 1 '''
    # Visualising data
    # Plot different visualizations of the data
    visualize_prices_across_neighborhood(manhattan_data)
    visualize_scatter_matrix(manhattan_data)
    visualise_regression_plot(manhattan_data)
    visualize_corr_matrix(manhattan_data)
    visualize_boxplot(manhattan_data)
    visualize_violinplot(manhattan_data)

    # Select predictor variables
    X = select_predictor_variables(manhattan_data)
    # Set the target variable
    y = manhattan_data['logprices']
    # Fit the linear model
    model = fit_linear_model(X, y)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Build the linear regression model on the training data
    lr_model = train_linear_regression_model(X_train, y_train)
    # Make predictions on the test set using the trained model
    y_pred = predict_test_data(lr_model, X_test)
    # Calculate the residuals as the difference between the actual values (y_test) and predicted values (y_pred):
    residuals = calculate_residuals(y_test, y_pred)
    # Calculate the cross-validation score for the linear regression model
    cv_scores = calculate_cv_scores(lr_model, X, y)
    # Print the mean cross-validation scores
    print("Mean CV Score:", cv_scores.mean())
    # print the mean squared score
    print(f"Final Model MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2_scores: {r2_score(y_test, y_pred)}")
    # Plot a histogram of the residuals.
    plot_residuals_histogram(residuals)

    ''' PART 2 Improved model '''
    # Read data and preprocess the data
    data_2 = preprocess_data_for_part_2(path)

    # Visualize the correlation matrix for the improved data
    visualize_corr_matrix(data_2)
    # Select features using Recursive Feature Elimination
    selected_features = select_features_with_rfe(data_2)
    selected_features_list = selected_features.tolist()

    print(f"The selected predictors choosen are: {selected_features_list}")

    # Perform Bayesian Optimization for hyperparameter tuning
    bayesian_Optimization(data_2, 'logprices', selected_features_list)
    # Calculate the silhouette score for clustering
    scores = silhouetteScore(data_2)
    # Draws the silhouette scores for justification
    visualize_silhouette_scores(scores)
    # Create a KMeans object with the following hyperparameters
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Fit the KMeans object to the data
    kmeans.fit(data_2.select_dtypes(include=np.number))

    # Visualize clusters with 'logprices' and 'year_built'
    visualize_clusters(data_2, kmeans.labels_, 'logprices', 'year_built', 'K-Means Clustering Results (K=2)',
                       'logprices', 'year_built')

    # Visualize clusters with 'logprices' and 'total_units'
    visualize_clusters(data_2, kmeans.labels_, 'logprices', 'total_units', 'K-Means Clustering Results (K=2)',
                       'logprices', 'total_units')

    # Visualize clusters with 'year_built' and 'gross_square_feet'
    visualize_clusters(data_2, kmeans.labels_, 'year_built', 'gross_square_feet', 'K-Means Clustering Results (K=2)',
                       'year_built', 'gross_square_feet')

    # Add cluster labels to the dataset and create separate dataframes for each cluster
    cluster_dataframes = create_cluster_dataframes(data_2, kmeans.labels_)

    # Split the data and train linear regression models for each cluster
    X_train_list, X_test_list, y_train_list, y_test_list, lr_models = split_and_train_clusters(cluster_dataframes)

    # Evaluate the models for each cluster
    mse_list, mae_list = evaluate_cluster_models(lr_models, X_test_list, y_test_list)

    # Print the evaluation results
    for i in range(len(lr_models)):
        print(f"Cluster {i}: Mean Squared Error = {mse_list[i]:.4f}, Mean Absolute Error = {mae_list[i]:.4f}")

    # Allows me to compare between to my regression model obtained in part 2.1 graphically.
    compare_models_graphically(data_2, lr_models, X_test_list, y_test_list)

# TODO 1- Box plot for average prices across neighborhoods - possibly using log proces instead of sale prices
# TODO - See if you can improve residual and predicted plot too understand why the data is in clusters
# TODO 2- average sale prices over time each month and Total units over time (done)
# TODO 3- regression plot (done)
# TODO 4- a graph to show correlation between numerical values (done)
# TODO 5- draw silhouette graph (done)
# TODO 6- show end cluster data
