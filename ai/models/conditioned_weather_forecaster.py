import pandas as pd
import numpy as np
from merlion.utils.time_series import TimeSeries
# Import models & configs
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.smoother import MSES, MSESConfig
# Import data pre-processing transforms
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
import json
import os
import pprint
from merlion.models.factory import ModelFactory
from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig, ForecastMetric


class kCP_Rain_Forecaster():
    
    def __init__(self, rain_data, time_field, measure, time_break, target_sampling, target_forecast_index):
        self.rain_data = rain_data
        self.time_field = time_field
        self.measure = measure
        self.time_break = time_break
        self.target_sampling = target_sampling
        self.target_forecast_index = target_forecast_index
        
    def preprocess_endog_features(self, features):
        self.rain_data[self.measure].fillna(0.0, inplace=True)
        self.rain_data['rain_label'] = np.array([self.rain_data[self.measure].values[i] > 0.0 for i in range(self.rain_data.shape[0])]).astype(int)
        for feature in features:
            mean_0 = self.rain_data[(~self.rain_data[feature].isnull()) & (self.rain_data.rain_label == 0)][feature].mean()
            mean_1 = self.rain_data[(~self.rain_data[feature].isnull()) & (self.rain_data.rain_label == 1)][feature].mean()
            m = self.rain_data[feature].isna()
            self.rain_data.loc[m, feature] = np.where(self.rain_data.loc[m, 'rain_label'].eq(0), mean_0, mean_1)
            
        rain_endog_reduced = self.rain_data.loc[:, [self.time_field] + features]
        rain_data_hourly = rain_endog_reduced.set_index(self.time_field).resample(self.target_sampling).mean()
        modified_rain_data = rain_data_hourly.dropna()
        
        return modified_rain_data
        
    
    def preprocess_exog_features(self, is_origin, priori_features):
        for feature in priori_features:
            self.rain_data[feature].fillna(0.0, inplace=True)
        if is_origin == True:
            rain_exog_reduced = self.rain_data.loc[:, [self.time_field] + priori_features]
            modified_rain_data = rain_exog_reduced.set_index(self.time_field)
        else:
            rain_exog_reduced = self.rain_data.loc[:, [self.time_field] + priori_features]
            modified_rain_data = rain_exog_reduced.set_index(self.time_field).resample(self.target_sampling).mean()
  
        return modified_rain_data
  
  
    def train_without_priori(self, full_data):
        train_data = full_data[full_data.index < self.time_break]
        test_data = full_data[full_data.index >= self.time_break]
        # Split the time series into train/test splits, and convert it to Merlion format
        train_data = TimeSeries.from_pd(train_data)
        test_data  = TimeSeries.from_pd(test_data)
        # Train a model without exogenous data
        model = Prophet(ProphetConfig(target_seq_index=self.target_forecast_index))
        model.train(train_data)
        pred, err = model.forecast(test_data.time_stamps)
        smape = ForecastMetric.sMAPE.value(test_data, pred, target_seq_index=model.target_seq_index)
        rmse = ForecastMetric.RMSE.value(test_data, pred, target_seq_index=model.target_seq_index)
        print(f"sMAPE (w/o priori) = {smape:.2f}")
        print(f"RMSE (w/o priori) = {rmse:.2f}")

        return model, train_data, test_data
        
    
    def train_with_priori(self, full_data, df_priori):
        train_data = full_data[full_data.index < self.time_break]
        test_data = full_data[full_data.index >= self.time_break]
        # Split the time series into train/test splits, and convert it to Merlion format
        train_data = TimeSeries.from_pd(train_data)
        test_data  = TimeSeries.from_pd(test_data)
        # Get the priori variables Y
        priori = TimeSeries.from_pd(df_priori)
        # Train a model with priori data
        priori_model = Prophet(ProphetConfig(target_seq_index=self.target_forecast_index))
        priori_model.train(train_data, exog_data=priori)
        priori_pred, priori_err = priori_model.forecast(test_data.time_stamps, exog_data=priori)
        priori_smape = ForecastMetric.sMAPE.value(test_data, priori_pred, target_seq_index=priori_model.target_seq_index)
        priori_rmse = ForecastMetric.RMSE.value(test_data, priori_pred, target_seq_index=priori_model.target_seq_index)
        print(f"sMAPE (w/ priori)  = {priori_smape:.2f}")
        print(f"RMSE (w/ priori)  = {priori_rmse:.2f}")

        return priori_model, train_data, test_data
        
        
    @staticmethod
    def save_model(model, priori_model) -> None:
        # Save the model
        os.makedirs("models", exist_ok=True)
        path_model = os.path.join("models", "without_priori")
        model.save(path_model)
        # Print the config saved
        pp = pprint.PrettyPrinter()
        with open(os.path.join(path_model, "config.json")) as f_model:
            print(f"{type(model).__name__} Config")
            pp.pprint(json.load(f_model))
            
        path_priori_model = os.path.join("models", "with_priori")
        priori_model.save(path_priori_model)
        pp = pprint.PrettyPrinter()
        with open(os.path.join(path_priori_model, "config.json")) as f_priori_model:
            print(f"priori model Config")
            pp.pprint(json.load(f_priori_model))
            
        return None
        
    @staticmethod
    def load_model(model_name, model_path):
        # Load the selector using the ModelFactory
        model_factory_loaded = ModelFactory.load(name=model_name, model_path=model_path)
        return model_factory_loaded
    
    @staticmethod    
    def create_evaluator(loaded_model):
        # Re-initialize the model, so we can re-train it from scratch
        loaded_model.reset()
    
        # Create an evaluation pipeline for the model, where we
        # -- get the model's forecast every hour
        # -- have the model forecast for a horizon of 6 hours
        # -- re-train the model every 12 hours
        # -- when we re-train the model, retrain it on only the past 2 weeks of data
        # evaluator = ForecastEvaluator(model=loaded_model, config=ForecastEvaluatorConfig(
        #                               cadence="1h", horizon="6h", retrain_freq="12h", train_window="14d"))
        evaluator = ForecastEvaluator(model=loaded_model, config=ForecastEvaluatorConfig(
                                      cadence="1h", horizon="6h", retrain_freq=None))
        return evaluator
    
    @classmethod
    def evaluate_without_priori(cls, loaded_model, train_data, test_data):
        evaluator = cls.create_evaluator(loaded_model)
        train_result, test_result = evaluator.get_predict(train_vals=train_data, test_vals=test_data) 
        smape = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.sMAPE)
        rmse = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.RMSE)
        mae = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.MAE)
        
        metrics = {'smape': smape, 'rmse': rmse, 'mae': mae}
        return train_result, test_result, metrics
        
        
    @classmethod
    def evaluate_with_priori(cls, loaded_model, df_priori, train_data, test_data):
        # Get the priori variables Y
        priori = TimeSeries.from_pd(df_priori)
        evaluator = cls.create_evaluator(loaded_model)
        train_result, test_result = evaluator.get_predict(train_vals=train_data, test_vals=test_data, exog_data=priori) 
        smape = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.sMAPE)
        rmse = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.RMSE)
        mae = evaluator.evaluate(ground_truth=test_data, predict=test_result, metric=ForecastMetric.MAE)
        
        metrics = {'smape': smape, 'rmse': rmse, 'mae': mae}
        return train_result, test_result, metrics
        
        