import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX


def calculate_metrics(actual: pd.Series, predicted: pd.Series):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mape


def sarima_forecast(df: pd.DataFrame, steps: int = 12):
    df = df.sort_index()
    train = df.iloc[:-steps]
    test = df.iloc[-steps:]
    model = SARIMAX(train['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    forecast = pd.Series(results.predict(start=test.index[0], end=test.index[-1]), index=test.index)
    rmse, mape = calculate_metrics(test['value'], forecast)
    combined = pd.concat([train['value'], test['value']], axis=0)
    forecast_full = pd.Series(index=combined.index, dtype=float)
    forecast_full.loc[test.index] = forecast
    return forecast_full, rmse, mape, combined


def main():
    st.title('SARIMA Sales Forecasting')
    st.write('Upload CSV files with `date` and `value` columns.')

    uploaded_files = st.file_uploader('Upload CSVs', type='csv', accept_multiple_files=True)
    metrics_rows = []

    if uploaded_files:
        for file in uploaded_files:
            st.header(f'File: {file.name}')
            df = pd.read_csv(file)
            if 'date' not in df.columns or 'value' not in df.columns:
                st.error('CSV must contain `date` and `value` columns.')
                continue

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            forecast_full, rmse, mape, series = sarima_forecast(df)

            plot_df = pd.DataFrame({'actual': series, 'forecast': forecast_full})
            fig = px.line(plot_df, x=plot_df.index, y=['actual', 'forecast'],
                          labels={'value': 'Value', 'index': 'Date'},
                          title=f'Actual vs Forecast - {file.name}')
            st.plotly_chart(fig, use_container_width=True)

            metrics_rows.append({'File': file.name, 'RMSE': rmse, 'MAPE': mape})

    if metrics_rows:
        st.subheader('Metrics')
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df)


if __name__ == '__main__':
    main()
