from datetime import datetime
import json
import os
import requests
import boto3
import pandas as pd
import numpy as np

EXTERNAL_DATA_DIR = os.path.abspath('../data/external')
EXTERNAL_DATA_BUCKET = 'chi-finesse-covid19-mortality-rate-external-data'
TIME_SERIES_REPO = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
TIME_SERIES_DATA_PATH = 'csse_covid_19_data/csse_covid_19_time_series/'

EXTERNAL_DATA_ROLE = 'finesse-external-data'

ignore_countries = [
    'Others',
    'Cruise Ship'
]

cpi_country_mapping = {
    'United States of America': 'US',
    'China': 'Mainland China'
}

wb_country_mapping = {
    'United States': 'US',
    'Egypt, Arab Rep.': 'Egypt',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    'China': 'Mainland China',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Korea, Rep.': 'Korea, South'
}

wb_covariates = [
    ('SH.XPD.OOPC.CH.ZS',
        'healthcare_oop_expenditure'),
    ('SH.MED.BEDS.ZS',
        'hospital_beds'),
    ('HD.HCI.OVRL',
        'hci'),
    ('SP.POP.65UP.TO.ZS',
        'population_perc_over65'),
    ('SP.RUR.TOTL.ZS',
        'population_perc_rural')
]


def make_dataset():
    '''
    Main routine that grabs all COVID and covariate data and
    returns them as a single dataframe that contains:

    * count of cumulative cases and deaths by country (by today's date)
    * days since first case for each country
    * CPI gov't transparency index
    * World Bank data on population, healthcare, etc. by country
    '''
    _get_external_data(EXTERNAL_DATA_BUCKET)

    all_covid_data = _get_latest_covid_timeseries()

    covid_cases_rollup = _rollup_by_country(all_covid_data['confirmed_global'])
    covid_deaths_rollup = _rollup_by_country(all_covid_data['deaths_global'])

    todays_date = covid_cases_rollup.columns.max()

    # Create DataFrame with today's cumulative case and death count, by country
    df_out = pd.DataFrame({'cases': covid_cases_rollup[todays_date],
                           'deaths': covid_deaths_rollup[todays_date]})

    _clean_country_list(df_out)
    _clean_country_list(covid_cases_rollup)

    # Add observed death rate:
    df_out['death_rate_observed'] = df_out.apply(
        lambda row: row['deaths'] / float(row['cases']),
        axis=1)

    # Add covariate for days since first case
    df_out['days_since_first_case'] = _compute_days_since_nth_case(
        covid_cases_rollup, n=1)

    # Add CPI covariate:
    _add_cpi_data(df_out)

    # Add World Bank covariates:
    _add_wb_data(df_out)

    # Add World Bank diabetes data:
    _add_wb_data_diabetes(df_out)

    # Drop any country w/o covariate data:
    num_null = df_out.isnull().sum(axis=1)
    to_drop_idx = df_out.index[num_null > 1]
    print('Dropping %i/%i countries due to lack of data' %
          (len(to_drop_idx), len(df_out)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)

    return df_out


def _get_external_data(bucket):
    files = set(os.listdir(EXTERNAL_DATA_DIR))
    session = boto3.Session(profile_name=EXTERNAL_DATA_ROLE)

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.all():
        file = obj.key
        if file not in files:
            bucket.download_file(file, os.path.join(EXTERNAL_DATA_DIR, file))


def _get_latest_covid_timeseries():
    ''' Pull latest time-series data from JHU CSSE database '''

    repo = TIME_SERIES_REPO
    data_path = TIME_SERIES_DATA_PATH

    all_data = {}
    for status in ['confirmed_US', 'confirmed_global', 'deaths_US', 'deaths_global', 'recovered_global']:
        file_name = 'time_series_covid19_%s.csv' % status
        all_data[status] = pd.read_csv(
            '%s%s%s' % (repo, data_path, file_name))

    return all_data


def _rollup_by_country(df):
    '''
    Roll up each raw time-series by country, adding up the cases
    across the individual states/provinces within the country

    :param df: Pandas DataFrame of raw data from CSSE
    :return: DataFrame of country counts
    '''
    gb = df.groupby('Country/Region')
    df_rollup = gb.sum()
    df_rollup.drop(['Lat', 'Long'], axis=1, inplace=True, errors='ignore')

    return _convert_cols_to_dt(df_rollup)


def _convert_cols_to_dt(df):

    # Convert column strings to dates:
    idx_as_dt = [datetime.strptime(x, '%m/%d/%y') for x in df.columns]
    df.columns = idx_as_dt
    return df


def _clean_country_list(df):
    ''' Clean up input country list in df '''
    # handle recent changes in country names:
    country_rename = {
        'Hong Kong SAR': 'Hong Kong',
        'Taiwan*': 'Taiwan',
        'Czechia': 'Czech Republic',
        'Brunei': 'Brunei Darussalam',
        'Iran (Islamic Republic of)': 'Iran',
        'Viet Nam': 'Vietnam',
        'Russian Federation': 'Russia',
        'Republic of Korea': 'South Korea',
        'Republic of Moldova': 'Moldova',
        'China': 'Mainland China'
    }
    df.rename(country_rename, axis=0, inplace=True)

    df.drop(ignore_countries, axis=0, inplace=True, errors='ignore')


def _compute_days_since_nth_case(df_cases, n=1):
    ''' Compute the country-wise days since first confirmed case

    :param df_cases: country-wise time-series of confirmed case counts
    :return: Series of country-wise days since first case
    '''
    date_first_case = df_cases[df_cases >= n].idxmin(axis=1)
    days_since_first_case = date_first_case.apply(
        lambda x: 0 if pd.isnull(x) else (df_cases.columns.max() - x).days)
    # Add 1 month for China, since outbreak started late 2019:
    if 'Mainland China' in days_since_first_case.index:
        days_since_first_case.loc['Mainland China'] += 30
    # Fill in blanks (not yet reached n cases) with 0s:
    days_since_first_case.fillna(0, inplace=True)

    return days_since_first_case


def _add_cpi_data(df_input):
    '''
    Add the Government transparency (CPI - corruption perceptions index)
    data (by country) as a column in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add CPI data to df_input in place
    '''
    cpi_data = pd.read_excel(
        os.path.join(EXTERNAL_DATA_DIR, 'CPI2019.xlsx'), skiprows=2)
    cpi_data.set_index('Country', inplace=True, drop=True)
    cpi_data.rename(cpi_country_mapping, axis=0, inplace=True)

    # Add CPI score to input df:
    df_input['cpi_score_2019'] = cpi_data['CPI score 2019']

    return df_input


def _add_wb_data(df_input):
    '''
    Add the World Bank data covariates as columns in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add World Bank data to df_input in place
    '''
    wb_data = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'world_bank_data.csv'),
        na_values='..')

    for (wb_name, var_name) in wb_covariates:
        wb_series = wb_data.loc[wb_data['Series Code'] == wb_name]
        wb_series.set_index('Country Name', inplace=True, drop=True)
        wb_series.rename(wb_country_mapping, axis=0, inplace=True)

        # Add WB data:
        df_input[var_name] = _get_most_recent_value(wb_series)


def _add_wb_data_diabetes(df_input):
    '''
    Add the World Bank diabetes as columns in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add World Bank diabetes data to df_input in place
    '''
    wb_data = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'world_bank_diabetes.csv'),
        na_values='""')
    wb_data.set_index('Country Name', inplace=True, drop=True)
    wb_data.rename(wb_country_mapping, axis=0, inplace=True)

    # Add WB data:
    df_input['population_perc_diabetic'] = _get_most_recent_value(wb_data)


def _get_most_recent_value(wb_series):
    '''
    Get most recent non-null value for each country in the World Bank
    time-series data
    '''
    ts_data = wb_series[wb_series.columns[3::]]

    def _helper(row):
        row_nn = row[row.notnull()]
        if len(row_nn):
            return row_nn[-1]
        else:
            return np.nan

    return ts_data.apply(_helper, axis=1)
