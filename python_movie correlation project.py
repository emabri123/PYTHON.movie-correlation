import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

df = pd.read_csv(r'C:\Users\Ema\Desktop\alex\_PORTFOLIO\PYTHON\movies.csv')

#postotak praznih redova u svakom stupcu
for col in df.columns: 
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, percent_missing))

#brisanje redova koji sadrže null vrijednosti
df = df.dropna()
#print(df.isnull().sum())

#izmjena tipa podatka
#print(df.dtypes)
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

#ispravak stupca "year"
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)


df = df.sort_values(by=['gross'], inplace=False, ascending=False)
#pd.set_option('display.max_rows', None)

#uklanjanje duplikata
df.drop_duplicates(subset=['name', 'released', 'country', 'budget'], keep=False)

#scatter plot - budget vs. gross

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()

#scatter plot - budget vs. gross sa 

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.ylim(0,)
plt.show()

#prikaz korelacije kroz "heatmap"

correlation_matrix = df.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

#kategoriziranje ne numeričkih tipova podataka

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

print(df_numerized)

#prikaz korelacije svih vrijabli (numeričkih i kategoričkih)

correlation_matrix = df_numerized.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

#sada možemo pogledati koji parovi varijabli su u visokoj korelaciji

correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
sorted_pairs = corr_pairs.sort_values()
high_corr = sorted_pairs[(sorted_pairs) > 0.5]
print(high_corr)

#varijable 'votes' i 'budegt' imaju najviši stupanj korelacije 






