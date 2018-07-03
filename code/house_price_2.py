from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

train_housing_dataframe = pd.read_csv("/Users/liuxinzhong/Desktop/house_price_2/all/train.csv", sep=",")
# train_housing_dataframe.fillna(value=0)

def preprocess_features(train_housing_dataframe):
    digit_column_names = []
    # selected_features = train_housing_dataframe[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
    selected_features = train_housing_dataframe[['MSSubClass', 'LotFrontage']]

    processed_features = selected_features.copy()
    # processed_features.fillna(np.float(0.0))
    # print(processed_features)
    # print(processed_features.iloc[0,0])
    print(processed_features.columns.size)
    print(processed_features.iloc[0].size)
    for i in range(processed_features.columns.size):
        for j in range(processed_features.shape[0]):
            # print(3333)
            # print(processed_features.iloc[i,j])
            print(processed_features.iloc[j,i].dtype)
            print(processed_features.iloc[j,i])
            if(np.isnan(processed_features.iloc[j,i])):
                processed_features.iloc[j,i] = np.float(0.0)
    processed_features.to_csv("/Users/liuxinzhong/Desktop/house_price_2/all/train_1.csv")
    return processed_features

def preprocess_targets(train_housing_dataframe):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["SalePrice"] = (
        train_housing_dataframe["SalePrice"] / 1000.0)
    return output_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def train_model(
    learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
            optimizer=my_optimizer
            )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(
        training_examples, 
        training_targets["SalePrice"], 
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples, 
            training_targets["SalePrice"], 
            num_epochs=1, 
            shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
    validation_examples, validation_targets["SalePrice"], 
            num_epochs=1, 
            shuffle=False)
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
      # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
                steps=steps_per_period,
            )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])


        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
            # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")
    return linear_regressor

def main():
    training_examples = preprocess_features(train_housing_dataframe.head(1200))
    training_examples.describe()

    training_targets = preprocess_targets(train_housing_dataframe.head(1200))
    training_targets.describe()

    validation_examples = preprocess_features(train_housing_dataframe.tail(260))
    validation_examples.describe()

    validation_targets = preprocess_targets(train_housing_dataframe.tail(260))
    validation_targets.describe()

    linear_regressor = train_model(
        learning_rate=0.3,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
if __name__ == "__main__":
    main()