import pandas as pd
import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class Rain_Detection():
    
    def __init__(self, rain_data, time_field, features, measure, target_sampling, lags):
        self.rain_data = rain_data
        self.time_field = time_field
        self.features = features
        self.measure = measure
        self.target_sampling = target_sampling
        self.lags = lags
        
    def preprocess(self):
        self.rain_data[self.measure].fillna(0.0, inplace=True)
        self.rain_data['rain_label'] = np.array([self.rain_data[self.measure].values[i] > 0.0 for i in range(self.rain_data.shape[0])]).astype(int) 
        for feature in self.features:
            mean_0 = self.rain_data[(~self.rain_data[feature].isnull()) & (self.rain_data.rain_label == 0)][feature].mean()
            mean_1 = self.rain_data[(~self.rain_data[feature].isnull()) & (self.rain_data.rain_label == 1)][feature].mean()
            m = self.rain_data[feature].isna()
            self.rain_data.loc[m, feature] = np.where(self.rain_data.loc[m, 'rain_label'].eq(0), mean_0, mean_1)
            
        rain_data_reduced = self.rain_data.loc[:, [self.time_field] + self.features]
        rain_label = self.rain_data.loc[:, [self.time_field] + ['rain_label']]
        rain_data_hourly = rain_data_reduced.set_index(self.time_field).resample(self.target_sampling).mean()
        rain_label_hourly = rain_label.set_index(self.time_field).resample(self.target_sampling).sum()
        modified_rain_data = rain_data_hourly.join(rain_label_hourly)
          
        for lag in self.lags:
            modified_rain_data[f'rain_label_shifted_{lag}'] = modified_rain_data.rain_label.shift(-lag)
            modified_rain_data[f'rain_label_shifted_{lag}'].fillna(modified_rain_data.rain_label[-(lag+1)], inplace=True)
        
        return modified_rain_data
        
    
    def train(self, full_data, time_break):
        models = {}
        train_data = full_data[full_data.index < time_break]
        test_data = full_data[full_data.index >= time_break]
        for lag in self.lags:
            y_train = train_data.iloc[:,(lag-1-len(self.lags))]
            X_train = train_data.iloc[:,:-(len(self.lags)+1)]
            # define search
            model = AutoSklearnRegressor(time_left_for_this_task=6*60, per_run_time_limit=60, n_jobs=8)
            # perform the search
            model.fit(X_train, y_train)
            models.update({f"model_lag_{lag}": model})

        return train_data, test_data, models
        
    def evaluate(self, test_data, models):
        metrics = {}
        X_test = test_data.iloc[:,:-(len(self.lags)+1)]
        for lag in self.lags:
            y_test = test_data.iloc[:,(lag-1-len(self.lags))]
            y_hat = abs(models[f"model_lag_{lag}"].predict(X_test))
            mae = mean_absolute_error(y_test, y_hat)
            mse = mean_squared_error(y_test, y_hat)
            mape = mean_absolute_percentage_error(y_test, y_hat)
            X_test_result = X_test.copy()
            X_test_result['y_hat'] = y_hat
            X_test_result['y_test'] = y_test
            X_test_result['y_test_label'] = np.array([X_test_result.y_test.values[i] > 40.0 for i in range(X_test_result.shape[0])]).astype(int) 
            X_test_result['y_hat_label'] = np.array([X_test_result.y_hat.values[i] > 40.0 for i in range(X_test_result.shape[0])]).astype(int)
            acc = accuracy_score(X_test_result.y_test_label, X_test_result.y_hat_label)
            metrics.update({f"evaluation_metrics_model_{lag}": {'MSE': mse,'MAPE': mape,'MAE': mae,'accuracy': acc}})

        return X_test, metrics 
        
    @staticmethod    
    def infer(target_lag, x_future, models):
        for key in models.keys():
            if target_lag == int(key.split('_')[-1]):
                y_hat = models[key].predict(x_future)
                x_future['forecasted_values'] = abs(y_hat)
                x_future['is_rain'] = np.array([x_future.forecasted_values.values[i] > 40.0 for i in range(x_future.shape[0])])

        return x_future.loc[:, ['forecasted_values', 'is_rain']]
        
