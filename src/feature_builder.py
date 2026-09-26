import pandas as pd

def DataFrame_Creation(loans, transactions, clients, credits, mortgages, ratings, macro_data, cohorts):
    cohorts["score_date"] = pd.to_datetime(cohorts["score_date"])
    loans["дата_начала_периода"] = pd.to_datetime(loans["дата_начала_периода"])
    loans.columns = loans.columns.str.replace({'дата_начала_периода':'start_date', 'просрочка_дней':'days_overdue'})

    loans = loans.loc[loans["days_overdue"] >= 90, ["ID", "start_date"]]
    first_default = loans.groupby("ID", as_index=False)["start_date"].min()
    cohorts = pd.merge(cohorts, first_default, how = 'left', on = ['ID'])
    cohorts = cohorts[cohorts["start_date"].isna() | (cohorts["start_date"] >= cohorts["score_date"])].copy()

    cohorts["target"] = (cohorts["start_date"].notna() & 
                        (cohorts["start_date"] < cohorts["score_date"] + pd.Timedelta(days=365))).astype(int)
    cohorts = cohorts.sort_values(['score_date', 'ID'])
    cohorts = cohorts.reset_index().drop(columns = ['index', 'start_date'])

    clients.columns = clients.columns.str.replace({'возраст':'age', 'семейное_положение':'marital_status', 
                                                   'наличие_иждивенцев':'dependents', 'дата_регистрации':'registration_date'})
    clients['registration_date'] = pd.to_datetime(clients['registration_date'], format='%Y-%m-%d')
    mortgages.columns = mortgages.columns.str.replace({'дата_открытия':'opening_date', 'наличие_ипотеки':'mortgage'})
    mortgages['opening_date'] = pd.to_datetime(mortgages['opening_date'], format='%Y-%m-%d')

    clients = pd.merge(clients, mortgages, on = 'ID', how = 'left')
    clients = clients.drop(columns = ['opening_date', 'registration_date']).copy()
    clients['mortgage'] = clients['mortgage'].fillna(0)
    clients['mortgage'] = clients['mortgage'].astype(int)

    credits.columns = credits.columns.str.replace({'доход':'income', 'сумма_кредита':'loan_amount'})
    clients = pd.merge(clients, credits, on = 'ID', how = 'left')
    cohorts = pd.merge(cohorts, clients, on = 'ID', how = 'left')

    # Считаю, что рейтинг актуален на 'date', сдвиг здесь не нужен, т.к. это рейтинг фактически за прошлый месяц
    ratings['date'] = pd.to_datetime(ratings['date'], format='%Y-%m-%d')
    ratings.columns = ratings.columns.str.replace({'date':'score_date', 'кредитный_рейтинг':'credit_rating'})
    cohorts = pd.merge(cohorts, ratings, on = ['ID', 'score_date'], how = 'left')

    # Макроэкномические показатели собираются за месяц и округляются до первой даты начала месяца, 
    # следовательно необходимо сделать сдвиг в них, чтобы не допустить утечки
    macro_data.columns = macro_data.columns.str.replace({'date':'score_date', 'учетная_ставка':'discount_rate', 
                                                         'уровень_безработицы' : 'unemployment_rate', 'инфляция':'inflation'})
    macro_data['score_date'] = pd.to_datetime(macro_data['score_date'], format='%Y-%m-%d')
    macro_data['score_date'] = (macro_data['score_date'].dt.to_period('M') - 1).dt.to_timestamp()

    transactions['date'] = pd.to_datetime(transactions['date'], format='%Y-%m-%d')
    transactions.columns = transactions.columns.str.replace({'date':'score_date', 'MCC_5300':'mcc_5300', 'MCC_5814':'mcc_5814',
                                                             'MCC_5812':'mcc_5812', 'MCC_5411':'mcc_5411', 'MCC_3990':'mcc_3990',
                                                             'MCC_5722':'mcc_5722', 'MCC_4900':'mcc_4900', 'MCC_другое':'mcc_other'})
    transactions['score_date'] = (transactions['score_date'].dt.to_period('M') - 1).dt.to_timestamp()
    transactions = pd.merge(transactions, macro_data, on = ['score_date'], how = 'left')
    cohorts = pd.merge(cohorts, transactions, on = ['ID', 'score_date'], how = 'left')

    # После соединения последний месяц - с пустыми значениями по транзакциям, а также по макропоказателям, 
    # что ожидаемо после сдвига. Удалим строки с пустыми значениями.
    return cohorts.dropna()

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
    for win in [2,3,4,7]:
        for col in ['spendings', 'credit_rating']:
            avgs = col + '_avg' + str(win-1)

            df[avgs] = df.groupby('ID')[col].transform(lambda x: x.rolling(window = win, min_periods=1).mean())
            df[avgs] = df[avgs] / df[col]

    #Сортировка (на всякий случай). Назначение индекса для дальнейшей работы. Удаление лишних признаков, которые заменили долями
    df = df.sort_values(['score_date', 'ID'])
    df = df.set_index('score_date')
    df = df.drop(columns = ['ID', 'income', 'loan_amount', 'mcc_5300', 'mcc_5814', 'mcc_5812', 'mcc_5411', 'mcc_3990', 
                            'mcc_5722', 'mcc_4900', 'mcc_other', 'spendings']).copy()
    return df
