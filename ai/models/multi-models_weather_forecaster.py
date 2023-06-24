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


class Rain_Forecaster():
    
    def __init__(self, rain_data, time_field, features, measure, target_forecast, target_sampling):
        self.rain_data = rain_data
        self.time_field = time_field
        self.features = features
        self.measure = measure
        self.target_sampling = target_sampling
        self.target_forecast = target_forecast
        
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
        modified_rain_data.dropna(inplace=True)
        
        return modified_rain_data
        
    
    def train(self, full_data, time_break):
        rain_target_forecast = full_data[[self.target_forecast]]
        train_data = rain_target_forecast[rain_target_forecast.index < time_break]
        test_data = rain_target_forecast[rain_target_forecast.index >= time_break]
        # Split the time series into train/test splits, and convert it to Merlion format
        train_data = TimeSeries.from_pd(train_data)
        test_data  = TimeSeries.from_pd(test_data)
        
        config1 = ArimaConfig(max_forecast_steps=100, order=(20, 1, 5), transform=TemporalResample(granularity="1h"))
        model1 = Arima(config1)
        config2 = ProphetConfig(max_forecast_steps=None, transform=Identity())
        model2 = Prophet(config2)
        config3 = MSESConfig(max_forecast_steps=100, max_backstep=60, transform=TemporalResample(granularity="1h"))
        model3 = MSES(config3)
        # The combiner here will simply take the mean prediction of the ensembles here
        ensemble_config = ForecasterEnsembleConfig(combiner=Mean(), models=[model1, model2, model3])
        ensemble = ForecasterEnsemble(config=ensemble_config)
        # The combiner here uses the sMAPE to compare individual models, and
        # selects the model with the lowest sMAPE
        selector_config = ForecasterEnsembleConfig(combiner=ModelSelector(metric=ForecastMetric.sMAPE))
        selector = ForecasterEnsemble(config=selector_config, models=[model1, model2, model3])
        
        print("\nTraining ensemble...")
        forecast_e, stderr_e = ensemble.train(train_data)
        print("\nTraining model selector...")
        forecast_s, stderr_s = selector.train(train_data)
        print("Done!")

        return ensemble, selector, train_data, test_data
        
    @staticmethod
    def save_model(ensemble, selector) -> None:
        # Save the model
        os.makedirs("models", exist_ok=True)
        path_ensemble = os.path.join("models", "ensemble")
        ensemble.save(path_ensemble)
        # Print the config saved
        pp = pprint.PrettyPrinter()
        with open(os.path.join(path_ensemble, "config.json")) as f_ensemble:
            print(f"{type(ensemble).__name__} Config")
            pp.pprint(json.load(f_ensemble))
            
        path_selector = os.path.join("models", "selector")
        selector.save(path_selector)
        pp = pprint.PrettyPrinter()
        with open(os.path.join(path_selector, "config.json")) as f_selector:
            print(f"Selector Config")
            pp.pprint(json.load(f_selector))
            
        return None
        
    @staticmethod
    def load_model(model_path):
        # Load the selector using the ModelFactory
        model_factory_loaded = ModelFactory.load(name="ForecasterEnsemble", model_path=model_path)
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
        evaluator = ForecastEvaluator(model=loaded_model, config=ForecastEvaluatorConfig(
                                      cadence="1h", horizon="6h", retrain_freq="12h", train_window="14d"))
        return evaluator
    
    @classmethod
    def evaluate(cls, loaded_model, train_data, test_data):
        ensemble_evaluator = cls.create_evaluator(loaded_model)
        ensemble_train_result, ensemble_test_result = ensemble_evaluator.get_predict(train_vals=train_data, test_vals=test_data) 
        smape = ensemble_evaluator.evaluate(ground_truth=test_data, predict=ensemble_test_result, metric=ForecastMetric.sMAPE)
        rmse = ensemble_evaluator.evaluate(ground_truth=test_data, predict=ensemble_test_result, metric=ForecastMetric.RMSE)
        mae = ensemble_evaluator.evaluate(ground_truth=test_data, predict=ensemble_test_result, metric=ForecastMetric.MAE)
        
        metrics = {'smape': smape, 'rmse': rmse, 'mae': mae}
        return ensemble_test_result, metrics
        
        