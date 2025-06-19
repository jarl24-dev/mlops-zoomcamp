#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

categorical = ['PULocationID', 'DOLocationID']
target = 'duration'

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

def read_data(year, month):
    url=f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()  

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def prepare_dictionaries(df):
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts

def apply_model(dicts):
    dv, model = load_model()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred

def save_results(df, y_pred, output_file):
    df_results = pd.DataFrame(y_pred, columns=[target])
    df_results['ride_id'] = df['ride_id']

    print(df_results['duration'].mean())

    #output_file = f'./output/yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    df_results.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def run():

    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    print(f'reading the data from {year}-{month}...')
    df = read_data(year, month)
    dicts = prepare_dictionaries(df)

    print(f'applying the model...')
    y_pred = apply_model(dicts)

    output_file = f'./output/yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    print(f'saving the result to {output_file}...')
    save_results(df, y_pred, output_file)

if __name__ == '__main__':
    run()