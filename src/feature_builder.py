import pandas as pd
import numpy as np

def DataFrame_Creation(loans, transactions, clients, credits, mortgages, ratings, macro_data, cohorts):
    cohorts["score_date"] = pd.to_datetime(cohorts["score_date"])
    loans["дата_начала_периода"] = pd.to_datetime(loans["дата_начала_периода"])
    loans = loans.rename(columns={'дата_начала_периода':'start_date', 'просрочка_дней':'days_overdue'})

    loans = loans.loc[loans["days_overdue"] >= 90, ["ID", "start_date"]]
    first_default = loans.groupby("ID", as_index=False)["start_date"].min()
    cohorts = pd.merge(cohorts, first_default, how = 'left', on = ['ID'])
    cohorts = cohorts[cohorts["start_date"].isna() | (cohorts["start_date"] >= cohorts["score_date"])].copy()

    cohorts["target"] = (cohorts["start_date"].notna() & 
                        (cohorts["start_date"] < cohorts["score_date"] + pd.DateOffset(months=12))).astype(int)
    cohorts = cohorts.sort_values(['score_date', 'ID'])
    cohorts = cohorts.reset_index().drop(columns = ['index', 'start_date'])

    clients = clients.rename(columns={'возраст':'age', 'семейное_положение':'marital_status', 
                                      'наличие_иждивенцев':'dependents', 'дата_регистрации':'registration_date'})
    clients['registration_date'] = pd.to_datetime(clients['registration_date'], format='%Y-%m-%d')

    mortgages = mortgages.rename(columns={'дата_открытия':'opening_date', 'наличие_ипотеки':'mortgage'})
    mortgages['opening_date'] = pd.to_datetime(mortgages['opening_date'], format='%Y-%m-%d')

    clients = pd.merge(clients, mortgages, on = 'ID', how = 'left')
    clients = clients.drop(columns = ['opening_date']).copy()
    clients['mortgage'] = clients['mortgage'].fillna(0)
    clients['mortgage'] = clients['mortgage'].astype(int)

    credits = credits.rename(columns={'доход':'income', 'сумма_кредита':'loan_amount'})
    clients = pd.merge(clients, credits, on = 'ID', how = 'left')
    cohorts = pd.merge(cohorts, clients, on = 'ID', how = 'left')

    # Считаю, что рейтинг актуален на 'date', сдвиг здесь не нужен, т.к. это рейтинг фактически за прошлый месяц
    ratings['date'] = pd.to_datetime(ratings['date'], format='%Y-%m-%d')
    ratings = ratings.rename(columns={'date':'score_date', 'кредитный_рейтинг':'credit_rating'})
    cohorts = pd.merge(cohorts, ratings, on = ['ID', 'score_date'], how = 'left')

    macro_data = macro_data.rename(columns={'date':'score_date', 'учетная_ставка':'discount_rate', 
                                            'уровень_безработицы' : 'unemployment_rate', 'инфляция':'inflation'})
    macro_data['score_date'] = pd.to_datetime(macro_data['score_date'], format='%Y-%m-%d')

    transactions['date'] = pd.to_datetime(transactions['date'], format='%Y-%m-%d')
    transactions = transactions.rename(columns={'date':'score_date', 'MCC_5300':'mcc_5300', 'MCC_5814':'mcc_5814',
                                                'MCC_5812':'mcc_5812', 'MCC_5411':'mcc_5411', 'MCC_3990':'mcc_3990',
                                                'MCC_5722':'mcc_5722', 'MCC_4900':'mcc_4900', 'MCC_другое':'mcc_other'})
    transactions['score_date'] = (transactions['score_date'].dt.to_period('M') + 1).dt.to_timestamp()
    cohorts = pd.merge(cohorts, macro_data, on = ['score_date'], how = 'left')
    cohorts = pd.merge(cohorts, transactions, on = ['ID', 'score_date'], how = 'left')

    # После соединения последний месяц - с пустыми значениями по транзакциям. Добавим флаг - первый месяц клиента, 
    # а пропуски заполним нулем
    cohorts['is_new_client'] = cohorts['mcc_5300'].isna().astype(int)
    cohorts = cohorts.fillna(0)

    mxdt = cohorts['score_date'].max() - pd.DateOffset(months=11)
    return cohorts[cohorts['score_date'] < mxdt].copy()

def DataFrame_Creation_New_Features(loans, transactions, clients, credits, mortgages, ratings, macro_data, cohorts):
    df = DataFrame_Creation(loans, transactions, clients, credits, mortgages, ratings, macro_data, cohorts)

    df['income_per_dep'] = df['income'] / (df['dependents'] + 1)
    df['loan_per_inc'] = df['loan_amount'] / df['income_per_dep']
    df['spendings'] = (df['mcc_5300'] + df['mcc_5814'] + df['mcc_5812'] + df['mcc_5411'] + 
                       df['mcc_3990'] + df['mcc_5722'] + df['mcc_4900'] + df['mcc_other'])

    df['mcc_5300_sh'] = df['mcc_5300'] / df['income_per_dep']
    df['mcc_5814_sh'] = df['mcc_5814'] / df['income_per_dep']
    df['mcc_5812_sh'] = df['mcc_5812'] / df['income_per_dep']
    df['mcc_5411_sh'] = df['mcc_5411'] / df['income_per_dep']
    df['mcc_3990_sh'] = df['mcc_3990'] / df['income_per_dep']
    df['mcc_5722_sh'] = df['mcc_5722'] / df['income_per_dep']
    df['mcc_4900_sh'] = df['mcc_4900'] / df['income_per_dep']
    df['mcc_other_sh'] = df['mcc_other'] / df['income_per_dep']
    # Для древа решений полезный признак, но для линейных моделей - вредный:
    df['spendings_sh'] = (df['spendings']) / df['income_per_dep']
    for win in [3,4,7]:
        for col in ['spendings', 'credit_rating']:
            avgs = col + '_avg' + str(win-1)

            df[avgs] = df.groupby('ID')[col].transform(lambda x: x.rolling(window = win, min_periods=1).mean())
            df[avgs] = df[avgs] / df[col].replace(0, np.nan)
            df[avgs] = df[avgs].fillna(0)

    df['spendings_dif1_sh'] = (df['spendings'] - df.groupby('ID')['spendings'].shift(1)) / df['spendings']
    df['spendings_dif1_sh'] = df['spendings_dif1_sh'].fillna(0)

    df['credit_rating_dif1_sh'] = (df['credit_rating'] - df.groupby('ID')['credit_rating'].shift(1)) / df['credit_rating']
    df['credit_rating_dif1_sh'] = df['credit_rating_dif1_sh'].fillna(0)

    df['spendings_up_count6'] = df.groupby('ID')['spendings'].transform(lambda x: 
                                                                        x.diff().gt(0).rolling(6, min_periods=1).sum())

    df['credit_rating_up_count6'] = df.groupby('ID')['credit_rating'].transform(lambda x: 
                                                                                x.diff().gt(0).rolling(6, min_periods=1).sum())

    df['total_months'] = (df['score_date'].dt.year  - df['registration_date'].dt.year) * 12 + (df['score_date'].dt.month - df['registration_date'].dt.month)

    #Сортировка (на всякий случай). Назначение индекса для дальнейшей работы. Удаление лишних признаков, которые заменили долями
    df = df.sort_values(['score_date', 'ID'])
    df = df.set_index('score_date')
    df = df.drop(columns = ['ID', 'income', 'loan_amount', 'mcc_5300', 'mcc_5814', 'mcc_5812', 'mcc_5411', 'mcc_3990', 
                            'registration_date', 'mcc_5722', 'mcc_4900', 'mcc_other', 'spendings'])
    return df.copy()
