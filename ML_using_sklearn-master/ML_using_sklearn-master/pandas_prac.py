import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
# xyz_web = {'day' : [1,2,3,4,5,6],'visitors' : [1000,700,6000,1000,400,350] ,'bounce_rate' :[20,20,23,15,10,34]}
# df=pd.DataFrame(xyz_web)

#slicing
# print(df.head(2))
# print(df.tail(2))

# merging
# df1 = pd.DataFrame({'hpi' : [80,90,70,60], 'int_rate' : [2,1,2,3], 'ind_gdp' : [50,45,45,67]},index=[2001,2002,2003,2004])
# df2 = pd.DataFrame({'hpi' : [80,90,70,60], 'int_rate' : [2,1,2,3], 'ind_gdp' : [50,45,45,67]},index=[2005,2006,2007,2008])
# merge = pd.merge(df1,df2, on='hpi')
# print(merge)

#joining
# df1 = pd.DataFrame({'int_rate' : [2,1,2,3], 'ind_gdp' : [50,45,45,67]},index=[2001,2002,2003,2004])
# df2 = pd.DataFrame({'low_tier_hpi' : [2,1,2,3], 'umeployment' : [50,45,45,67]},index=[2001,2003,2004,2004])
# joined = df1.join(df2)
# print(joined)

#changing the index
# df = pd.DataFrame({'day' : [1,2,3,4],'visitors' : [200,100,230,300], 'bounce_rate' : [20,45,60,10]})
# df.set_index('day',inplace=True)
# # print(df)
# df.plot()
# plt.show()

#concatenation
# df1 = pd.DataFrame({'hpi' : [80,85,88,85],'int_rate' : [2,3,2,2],
#                     'us_gdp_thousands' : [50,55,65,55],},
#                    index=[2001,2002,2003,2004])
# df2 = pd.DataFrame({'hpi' : [80,85,88,85],'int_rate' : [2,3,2,2],
#                     'us_gdp_thousands' : [50,55,65,55],},
#                    index=[2005,2006,2007,2008])
# concat = pd.concat([df1,df2])
# print(concat)

#data conversion
# path = pd.read_csv('C:\\Users\\ASUS\\Downloads\\acme.csv',index_col=0)
# path.to_html('acme.html')

#example
country = pd.read_csv('C:\\Users\\ASUS\\Downloads\\data.csv',index_col=0)
df = country.head(5)
df = df.set_index('country_name',inplace=True)
sd = df.reindex(columns =['country_name','population'])
db = sd.diff(axis = 1)
db.plot(kind='bar')
plt.show()