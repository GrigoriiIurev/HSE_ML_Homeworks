import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="base"):
        self.mode = mode
        self.body_type_dict = {
            "sedan": "sedan", "sd": "sedan", "saloon": "sedan", "classic": "sedan",
            "hatchback": "hatchback", "hatch": "hatchback", "hb": "hatchback", 
            "back": "hatchback", "sportback": "hatchback", "3dr": "hatchback", "5dr": "hatchback",
            "wagon": "wagon", "estate": "wagon", "touring": "wagon", "tourer": "wagon",
            "sw": "wagon", "kombi": "wagon", "combi": "wagon", "variant": "wagon",
            "suv": "suv", "crossover": "suv", "cross": "suv", "4x4": "suv",
            "4wd": "suv", "awd": "suv", "jeep": "suv", "x": "suv",
            "coupe": "coupe", "coupé": "coupe", "cp": "coupe", "sport coupe": "coupe",
            "convertible": "convertible", "cabriolet": "convertible", "cabrio": "convertible",
            "roadster": "convertible", "spider": "convertible", "spyder": "convertible",
            "targa": "convertible", "barchetta": "convertible",
            "pickup": "pickup", "pick-up": "pickup", "truck": "pickup",
            "double cab": "pickup", "single cab": "pickup", "dc": "pickup",
            "sc": "pickup", "ute": "pickup",
            "mpv": "mpv", "minivan": "mpv", "van": "mpv", "touran": "mpv", "grand": "mpv",
            "liftback": "liftback", "fastback": "liftback", "gran coupe": "liftback",
            "micro": "micro", "kei": "micro", "keicar": "micro", "city": "micro",
            "panel van": "commercial", "cargo": "commercial", "delivery": "commercial",
            "commercial": "commercial", "cv": "commercial",
            "notchback": "notchback", "shooting brake": "shooting_brake",
            "landaulet": "landaulet", "phaeton": "phaeton",
            "limousine": "limousine", "brake": "shooting_brake"
        }
        
        self.transmission_map = {
            "AT": "AT", "A/T": "AT", "AUTOMATIC": "AT",
            "AMT": "AMT",
            "MT": "MT", "M/T": "MT", "MANUAL": "MT",
            "CVT": "CVT",
            "DCT": "DCT",
            "DSG": "DSG"
        }
        self.trim_levels = {
            "LXI", "VXI", "ZXI", "ZDI", "VDI", "LSI", "LDI", "GLS", "GXI",
            "VLS", "DLX", "CLX", "SXI", "LXi", "Vxi", "Zxi", "Zdi", "Vdi",
            "Sportz", "Classic", "Ambition", "Elegance", "Trend", "Style",
            "Dynamic", "Premium", "Exclusive", "Base", "Plus", "Optional"
        }

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()

        required_cols = [
            "name", "year", "km_driven", "fuel", "seller_type",
            "transmission", "owner", "mileage", "engine",
            "max_power", "torque", "seats"
        ]
        missing = [c for c in required_cols if c not in X.columns]

        if missing:
            raise ValueError(f"Не хватает колонок: {missing}")
        
        # Это все я использую так же в EDA!!!!

        # очищаю пробег, двигатель и мощность в train
        X['mileage'] = X['mileage'].apply(lambda x: str(x).split()[0] if pd.notna(x) else x).astype(float)
        X['engine'] = X['engine'].apply(lambda x: str(x).split()[0] if pd.notna(x) else x).astype(float)
        X['max_power'] = X['max_power'].apply(lambda x: self._clearing_max_power(x))

        # достаю числа и слова из torque и сохраняю во временные столбцы
        X[['torque_numbers', 'torque_words']] = (
            X['torque'].apply(lambda x: pd.Series(self._split_dig_words(x)))
        )

        # создаю чистый момент силы
        X['torque_clean'] = X.apply(
            lambda x: self._creat_torque_col(x['torque_numbers'], x['torque_words']),
            axis=1
        )

        # создаю максимальные обороты
        X['max_torque_rpm'] = (
            X['torque_numbers'].apply(lambda x: self._creat_max_torque_rpm_col(x))
        ).astype(float)
        X['torque'] = X['torque_clean']

        X.drop(columns=['torque_numbers', 'torque_words', 'torque_clean'], inplace=True)

        X['engine'] = X['engine'].astype(int)
        X['seats'] = X['seats'].astype(int)

        if self.mode in ["medium", "full"]:
            X['mark'] = X['name'].str.split(' ').str[0]
            X['model'] = X['name'].str.split(' ').str[1]

            X["body_type"] = X["name"].apply(lambda x: self._extract_body_type(x))
            X["parsed_transmission"] = X["name"].apply(lambda x: self._extract_transmission(x))
            X["parsed_engine_volume"] = X["name"].apply(self._extract_engine_volume)
            X["parsed_trim"] = X["name"].apply(lambda x: self._extract_trim(x))

            if self.mode == "full":
                X['age'] = 2020 - X['year']
                X['age_sq'] = X['age'] ** 2
                X['many_owners'] = (X['owner'].isin(['3rd Owner', '4th & Above Owner'])).astype(int)
                X['power_engine'] = X['max_power'] * X['engine']
                X['power_age'] = X['max_power'] * X['age']
                X["seat_int"] = X["seats"].astype(int)
                X["seat_obj"] = X["seats"].astype(str)
        return X

    def _split_dig_words(self, text):
        text = str(text).lower()
        text = text.replace(",", "")
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        words = re.findall(r'[a-zA-Z]+', text)
        return numbers, words
    
    def _creat_torque_col(self, numbers, words):
        if len(numbers) == 0:
            return np.nan
        if len(words) == 0:
            return float(numbers[0])
        if "kgm" in words:
            return float(numbers[0]) * 9.80665
        return float(numbers[0])
    
    def _creat_max_torque_rpm_col(self, numbers):
        if len(numbers) <= 1:
            return np.nan
        else:
            return max(numbers[1:])
        
    def _clearing_max_power(self, text):
        text = str(text).replace(",", "")
        number = re.search(r'\d+(\.\d+)?', str(text))
        return float(number.group()) if number else np.nan
    
    def _extract_transmission(self, name):
        for w in str(name).upper().split():
            if w in self.transmission_map:
                return self.transmission_map[w]
        return "unknown"

    def _extract_engine_volume(self, text):
        match = re.search(r"(\d\.\d)\s*L", text.upper())
        if match:
            return float(match.group(1))
        
        match = re.search(r"(\d\.\d)", text)
        if match:
            return float(match.group(1))

        match = re.search(r"(\d)\s*L", text.upper())
        if match:
            return float(match.group(1))
        
        return "unknow"

    def _extract_trim(self, name):
        for w in name.replace("-", " ").split():
            if w in self.trim_levels:
                return w
        return "unknown"

    def _extract_body_type(self, name):
        tokens = str(name).lower().split()
        for t in tokens:
            if t in self.body_type_dict:
                return self.body_type_dict[t]
        return "unknown"