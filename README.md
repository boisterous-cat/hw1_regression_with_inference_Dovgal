# HW1_Dovgal

# Результаты
## Предварительная обработка и анализ данных
* Первый этап:
    На первом этапе очистим данные mileage, engine, max_power от единиц измерения, чтобы можно было их преобразовать к числовому типу.
   Затем займемся колонкой torque. Здесь в одной строке есть два значения с разными единицами измерения, которые еще и отличаются от строки к строке. Для этого пишется несколько функций, который преобразуют колонку в torque и max_torque_rpm  с конвертированными данными для различных единиц измерения.
  Удалим дубликаты строк, но только те, в которых совпадают значения кроме цены.

* Второй этап:
    Пропуски заполним медианными значениями.

* Третий этап:
    Визуализируем данные. Визуализация показала что имеются выбросы и есть корреляция предиктора max_power с таргетом. Mileage и engine также скоррелированы между собой. Убедились в схожести распределений для тестового и обучающего набора данных.
  Также убедились в том, что таргет распределен неравномерно и имеет выбросы.

## Построение наивной модели
* Модель только на вещественных признаках:
    Обучена классическая линейная регрессия, проведена стандартизация фич. Получаем результат:
    * Оценка R2: 0.6004296535378724
    * Оценка MSE: 229684445132.77765
    Наиболее информативным оказался признак max_power.  
    Обучена Lasso-регрессия:
    * Оценка R2: 0.6004283699058307
    * Оценка MSE: 229685183001.13013
    Регуляризация не занулила веса.  
    Перебором по сетке подобраны оптимальные параметры для Lasso-регрессии.
    Получаем: {'alpha': 0.06}  
    * Оценка R2: 0.6004295774029268
    * Оценка MSE: 229684488897.31845  
    Веса не занулились.  
    Перебором по сетке подобраны оптимальные параметры для ElasticNet-регрессии.
    Получим: {'alpha': 0.0005, 'l1_ratio': 0.5}    
    * Оценка R2: 0.6003699560167202  
    * Оценка MSE: 229718760972.63675  
 * Добавляем категориальные фичи:
    Осуществлено OneHot-кодирование, проведена стандартизация фич и подобраны оптимальные параметры для гребневой регрессии.
    Получим:{'alpha': 0.9500000000000001}    
    * Оценка R2: 0.6401912878826219  
    * Оценка MSE: 206828322292.5613

## Feature Engineering
* Реализуем следующие функции:
    1. Из названия автомобиля получим имя бренда
    2. Логарифмируем selling_price.
    3. Удалим выбросы из selling_price и km_driven.
    4. Вместо признака year сделаем признак age - возраст авто.
    5. Закодируем владельца авто - owner.
    
    Подберем параметры для Ridge: {'alpha': 0.9500000000000001}
    Получим:
    Оценка R2: 0.8977027242408053
    Оценка MSE: 0.07472497494212138
    
    Наибольшее улучшение показало логарифмирование целевой переменной и удаление выбросов.  
  Соответствующую модель записываем в файл pickle.
    Далее строим "Бизнесовую" метрику и получаем что доля предиктов, отличающихся не более чем на 10% - 99.73%. 

## Реализация сервиса на FastAPI
* Реализуем следующее:
1. средствами pydantic описываем класс базового объекта
2. класс с коллецией объектов
3. метод post, который получает на вход один объект описанного класса
4. метод post, который получает на вход коллекцию объектов описанного класса

Получаем (скриншоты и гиф имееют разные примеры):  
1. Предсказание по одноу объекту:
      ![item_prediction2](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/e8af628f-a205-46f5-94e2-2be8ef698769)  

![item_prediction](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/fd166c54-72cf-4575-9ef7-8f273a902ba5)  
3. Предсказание по нескольким объектам:  
  ![json_prediction](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/ef901710-2a4b-497e-9ed8-734dd68f115d)

   ![items_prediciton](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/d4953673-5535-4c44-af4c-6f81c6c44033)  
4. Предсказания по csv файлу:
     ![csv_preediction](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/655fec10-478f-44b3-9c8a-db1085d23df5)

   ![csv_prediction](https://github.com/boisterous-cat/hw1_regression_with_inference_Dovgal/assets/93883573/f65a196a-671d-4e98-850e-5f69aaf2c3e4)



