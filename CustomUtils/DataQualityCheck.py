import pandas as pd

def data_quality_report(input_df:pd.DataFrame, type='df'):
    '''
    Function to report some key metrics on the dataset
    '''
    if type=='path':
        df = pd.read_csv(input_df)
    elif type == 'df':
        df = input_df
    else:
        return 'Function param missing or incorrect. (type="df" or "path")'

    columns = df.columns
    shape = df.shape

    null_vals = df.isnull().sum()
    duplicates = df.duplicated().sum()

    # outliers
    num_columns = df.select_dtypes(include='number').columns
    outliers = {}
    description = df.describe()
    for col in num_columns:
        upper = description[col].loc['75%'] + 1.5*(description[col].loc['std'])
        lower = description[col].loc['25%'] - 1.5*(description[col].loc['std'])

        outliers[col] = ((df[col] > upper) | (df[col] < lower)).sum()
    
    # IQR
    iqr = description.loc['75%'] - description.loc['75%']


    # write to a text file
    with open('Data_Quality_Report.txt', 'w') as f:
        f.write('DATA QUALITY REPORT\n')
        f.write('=' * 50 + '\n\n')
        
        f.write(f'Shape: {shape}\n')
        f.write(f'Columns: {list(columns)}\n\n')
        
        f.write('Null Values:\n')
        f.write(str(null_vals) + '\n\n')
        
        f.write(f'Duplicate Rows: {duplicates}\n\n')
        
        f.write('Outliers (per column):\n')
        for col, count in outliers.items():
            f.write(f'  {col}: {count}\n')
        f.write('\n')
        
        f.write('IQR (Interquartile Range):\n')
        f.write(str(iqr) + '\n')

        f.write('Completed')

    return None