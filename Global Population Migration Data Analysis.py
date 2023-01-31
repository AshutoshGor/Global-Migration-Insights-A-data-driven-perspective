# Author : Ashutosh Laxminarayan Gor
# Importing all the required libraries:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file
import seaborn as sns

# Extracting data-set into dataframe:
df = pd.read_csv("UN_Migrations.csv", header = 1)


# Cleaning the data-set for specific data requirement:
df = df.drop(columns=df.columns[[1,3,4,5]])
df.rename( columns={'Unnamed: 0':'Year', 'Unnamed: 2' : 'Destination'}, inplace=True)

df.dropna(how='all', axis=1, inplace=True)
df.dropna(how='all', axis=0, inplace=True)

df = df.fillna(0)
df = df.replace(',','', regex=True)

# Converting data-set type of years to integer:
df.iloc[:,0] = df.iloc[:,0].astype(int)

# Capturing the required data-set:
df1 = df.loc[(df['Destination'] == "WORLD" ) |
             (df['Destination'] == "More developed regions") |
             (df['Destination'] == "Less developed regions") |
             (df['Destination'] == "High-income countries") |
             (df['Destination'] == "Low-income countries") |
             (df['Destination'] == "Northern America")]
df1.reset_index(drop=True, inplace=True)

# Converting data type for years data-set column:
df1['Year'] = df1.Year.astype(int)

col1 = len(df1.iloc[:,2:].columns)

# Converting data type for all columns of the data-set:
for i in range(2,(col1+2)):
    col = df1.iloc[:,i]
    col = [s.replace(",", "") for s in col]
    col = np.array(col).astype(int)
    df1.iloc[:,i] = col


# Line Plot for population migration across the world:
print('Total population migration around the world:')
df_world = df1.loc[(df1['Destination'] == 'WORLD')]
df_world.reset_index(drop=True, inplace=True)
df_world = df_world.iloc[: , :3]
print(df_world)

x = df_world['Year']
y = [num/1000000 for num in df_world['Total']]
line_plot = figure(title="Total population migration around the World", x_axis_label="Years",
                   y_axis_label="Population Migration (in Millions)")
line_plot.line(x, y, color='black', line_width=2)
show(line_plot)
print('\n')


# Bar Graph for population migration from low to high income countries:
df_INCOME = df1.loc[(df1['Destination'] == 'High-income countries') |
                    (df1['Destination'] == 'Middle-income countries') |
                    (df1['Destination'] == 'Upper-middle-income countries') |
                    (df1['Destination'] == 'Lower-middle-income countries') |
                    (df1['Destination'] == 'No income group available') |
                    (df1['Destination'] == 'Low-income countries')]

df_INCOME = df_INCOME.iloc[: , :3]
df_INCOME.reset_index(drop=True, inplace=True)


# Bar graph for population migration from low to high-income countries:
print('Total population migration to High-income countries:')
df_INCOME_high = df_INCOME.loc[(df_INCOME['Destination'] == "High-income countries")]
df_INCOME_high.reset_index(drop=True, inplace=True)
print(df_INCOME_high)

x = list(df_INCOME_high["Year"].astype(str))
y = list(num/1000000 for num in df_INCOME_high["Total"])

sns.barplot(x = x,y = y,data = df_INCOME)
plt.title("Population migration to High-income Countries")
plt.xlabel("Years")
plt.ylabel("Population (in Millions)")
plt.show()
print('\n')

# Bar graph migration from high to low income countries:
print('Total population migration to Low-income countries:')
df_INCOME_low = df_INCOME.loc[(df_INCOME['Destination'] == "Low-income countries")]
df_INCOME_low.reset_index(drop=True, inplace=True)
print(df_INCOME_low)

x = list(df_INCOME_low["Year"].astype(str))
y = [num/1000000 for num in df_INCOME_low["Total"]]

sns.barplot(x = x,y = y,data = df_INCOME)
plt.title("Population migration to Low-income Countries")
plt.xlabel("Years")
plt.ylabel("Population Migration (in Millions)")
plt.show()
print('\n')

# Capturing required data-set for population migration in More developed & Less developed regions:
df_develop = df1.loc[(df1['Destination'] == "More developed regions") |
                     (df1['Destination'] == "Less developed regions") |
                     (df1['Destination'] == "Least developed countries") |
                     (df1['Destination'] == "Less developed regions excluding least developed countries")]
df_develop = df_develop.iloc[: , :3]


# Bar graph for population migration to More developed regions:
print('Total population migration to More developed regions:')
df_develop_more = df_develop.loc[(df_develop['Destination'] == "More developed regions")]
df_develop_more.reset_index(drop=True, inplace=True)
print(df_develop_more)

x = list(df_develop_more["Year"].astype(str))
y = [num/1000000 for num in df_develop_more["Total"]]

sns.barplot(x = x,y = y,data = df_develop)
plt.title("Population migration to more developed regions:")
plt.xlabel("Years")
plt.ylabel("Population Migration (in Millions)")
plt.show()
print('\n')

# Bar graph migration to Less developed regions:
print('Total population migration to Less developed regions:')
df_develop_less = df_develop.loc[(df_develop['Destination'] == "Less developed regions")]
df_develop_less.reset_index(drop=True, inplace=True)
print(df_develop_less)

x = list(df_develop_less["Year"].astype(str))
y = [num/1000000 for num in df_develop_less["Total"]]

sns.barplot(x = x,y = y,data = df_develop)
plt.title("Population migration to Less developed regions:")
plt.xlabel("Years")
plt.ylabel("Population Migration (in Millions)")
# plt.show()


# Capturing data for top 10 countries with highest migration to Northern America:
df_america = df1.loc[(df1['Destination'] == 'Northern America')]
df_america.reset_index(drop=True, inplace=True)
df_america = df_america.set_index(['Year'])
df_america.drop(df_america.columns[[0,1]], axis=1, inplace=True)
df_america = df_america.transpose()


# plotting pie chart for Top 10 countries with major population migration to Northern America:
print('\nTop 10 countries with major population migration to Northern America:')
df_america_all1 = df_america[df_america.columns[0:1]]
df_america_largest1 = df_america_all1.nlargest(10, [1990])
print(df_america_largest1)
plot1 = df_america_largest1.plot.pie(y=1990, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot1.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot1.set_title('Top 10 countries with major population migration to Northern America in 1990', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all2 = df_america[df_america.columns[1:2]]
df_america_largest2 = df_america_all2.nlargest(10, [1995])
print(df_america_largest2)
plot2 = df_america_largest2.plot.pie(y=1995, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot2.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot2.set_title('Top 10 countries with major population migration to Northern America in 1995', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all3 = df_america[df_america.columns[2:3]]
df_america_largest3 = df_america_all3.nlargest(10, [2000])
print(df_america_largest3)
plot3 = df_america_largest3.plot.pie(y=2000, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot3.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot3.set_title('Top 10 countries with major population migration to Northern America in 2000', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all4 = df_america[df_america.columns[3:4]]
df_america_largest4 = df_america_all4.nlargest(10, [2005])
print(df_america_largest4)
plot4 = df_america_largest4.plot.pie(y=2005, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot4.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot4.set_title('Top 10 countries with major population migration to Northern America in 2005', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all5 = df_america[df_america.columns[4:5]]
df_america_largest5 = df_america_all5.nlargest(10, [2010])
print(df_america_largest5)
plot5 = df_america_largest5.plot.pie(y=2010, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot5.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot5.set_title('Top 10 countries with major population migration to Northern America in 2010', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all6 = df_america[df_america.columns[5:6]]
df_america_largest6 = df_america_all6.nlargest(10, [2015])
print(df_america_largest6)
plot6 = df_america_largest6.plot.pie(y=2015, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot6.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot6.set_title('Top 10 countries with major population migration to Northern America in 2015', fontsize=15, fontweight='bold')
plt.show()
print('\n')

df_america_all7 = df_america[df_america.columns[6:7]]
df_america_largest7 = df_america_all7.nlargest(10, [2019])
print(df_america_largest7)
plot7 = df_america_largest7.plot.pie(y=2019, autopct='%1.1f%%', shadow=False, legend=True, ylabel='', labeldistance=None,
                                     figsize=(10, 10), fontsize = 15)
plot7.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plot7.set_title('Top 10 countries with major population migration to Northern America in 2019', fontsize=15, fontweight='bold')
plt.show()
