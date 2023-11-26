from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from io import BytesIO, StringIO
import pandas as pd
import pickle
import gzip
import numpy as np
import random
import re
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

with gzip.open('pickle_model.pkl', 'rb') as ifp:
    MODEL = pickle.load(ifp)

def preprocessing(df_test):
    '''
    Здесь будет преобразование и очистка данных от лишнего мусора, например колонка torque

    :param df_test: исходный датафрей
    :return: df_test: очищенный датафрейм
    '''
    #1 Уберем единицы измерения из колонок mileage, engine, max_power
    cols = ['mileage', 'engine', 'max_power']
    for i in range(len(cols)):
        df_test[cols[i]] = df_test[cols[i]].astype(str).str.extract('(\d*\.\d+|\d+)').astype(float)
    #2 займемся колонкой torque
    def get_nums(s):
        '''
        Найдем все числовые вхождения в строку
        Запятую удалим
        Args:
              s(str): Строка

          Returns:
              res(List<str>): Список строк
        '''
        s = s.replace(',', '')
        res = re.findall(r"\d*\.?\d+|\d+", s)
        return res

    def get_torque(s):
        '''
        Преобразуем найденные значения.
        Для значений kgm переводим в np.

        Args:
              s(str): Строка

          Returns:
              (float/nan): Значение torque
        '''
        try:
            res = get_nums(s)
            if ('kgm' in s):
                return round(9.80665 * float(res[0]), 2)
            else:
                return float(res[0])
        except:
            return np.nan

    def get_rpm(s):
        '''
        Преобразуем найденные значения rpm.
        Так как мы ищем max_torque_rpm, то будем брать максимальное значение для тех случаев,
        когда в полученной строке больше 2 значений

        Args:
              s(str): Строка

          Returns:
              (float/nan): Значение rpm
        '''
        try:
            res = get_nums(s)
            if (len(res) > 2):
                rpm = max(res[1::], key=lambda x: float(x))
            else:
                rpm = float(res[1])
            return rpm
        except:
            return np.nan

    df_test['max_torque_rpm'] = df_test['torque'].apply(get_rpm).astype(float)
    df_test['torque'] = df_test['torque'].apply(get_torque)

    #3 заполним пропуски средним значением, если они есть
    nan_columns = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
    for i in range(len(nan_columns)):
        df_test[nan_columns[i]].fillna(df_test[nan_columns[i]].median(skipna=True), inplace=True)

    #4 получим имя бренда
    companyName = df_test['name'].apply(lambda x: x.split(' ')[0])
    df_test.insert(3, "company_name", companyName)
    df_test.drop(['name'], axis=1, inplace=True)

    #5 закодиуем владельца
    df_test['owner'] = df_test['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3})
    df_test['owner'] = pd.to_numeric(df_test['owner'], errors='coerce')

    #6 получим возраст авто вместо года выпуска
    df_test['age'] = datetime.date.today().year - df_test['year']
    df_test.drop(['year'], axis=1, inplace=True)

    #7 Преобразуем к float
    cols = ['km_driven', 'engine', 'age', 'seats']
    for i in range(len(cols)):
        df_test[cols[i]] = df_test[cols[i]].astype(float)

    #8 логарифмируем целевую переменную
    df_test['selling_price'] = np.log(df_test['selling_price'])

    return df_test

def predict_by_df(df_test):

    #float_columns = ['engine', 'seats']
    #for i in range(len(float_columns)):
        #df_test[float_columns[i]] = df_test[float_columns[i]].astype(int)
    df_test = preprocessing(df_test)

    predictions = np.exp(MODEL.predict(df_test.dropna()))
    return predictions

@app.post("/predict_item", summary='Get predicitions for item')
def predict_item(item: Item) -> float:
    '''
    Получаем объект с данными по машине.
    Отправляем за результатами в функцию predict_by_df.

    :param item: объект класса Item
    :return: полученная цена продажи авто.
    '''
    predicted = predict_items([item])
    return predicted[0] if len(predicted) else None

@app.post("/predict_items", summary='Get predicitions for json')
def predict_items(items: List[Item]) -> List[float]:
    '''
    Получаем json, создаем из него датафрейм и отправляем за результатами в функцию predict_by_df.

    :param items: список в формате json
    :return: список предсказанных цен по каждому объекту
    '''
    return [float(i) for i in predict_by_df(pd.DataFrame(jsonable_encoder(items)))]

@app.post("/csv_content", summary='Get predicitions for csv')
def get_csv(file: UploadFile):
    '''
    Получаем файл, считываем его в датафрейм.
    Отправляем датафрейм в функцию predict_by_df.
    Получаем файл с предсказаниями.

    :param file: загружаемый файл csv
    :return: файл с предсказаниями
    '''
    content = file.file.read()
    buffer = BytesIO(content)
    df=pd.read_csv(buffer, index_col=0)
    buffer.close()
    file.close()
    df['predict']=pd.Series(predict_by_df(df))
    df.to_csv('predictions.csv')
    response = FileResponse(path='predictions.csv', media_type='text/csv', filename='predictions.csv')
    return response