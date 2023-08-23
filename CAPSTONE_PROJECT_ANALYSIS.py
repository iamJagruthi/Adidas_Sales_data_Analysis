#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path = "E:/jagruthi_DA/Adidas_US_Sales_Datasets_cleaned.csv"
df = pd.read_csv(file_path)
print(df.head())


# In[3]:


#removing duplicates 
df = df.drop_duplicates()
df


# In[4]:


#calculating nulls
df.isnull().sum()


# In[5]:


#changing the date formats
import datetime
from dateutil.parser import parse
df['Invoice Date'] = df['Invoice Date'].apply(lambda my_date:parse(my_date).strftime('%d/%m/%y'))
print(df)


# In[6]:


#changing the column names
df.rename(columns={'Retailer ID':'Retailer_ID','Invoice Date':'Invoice_Date','Price per Unit':'Price_per_Unit','Units Sold':'Units_Sold','Total Sales':'Total_Sales','Operating Profit':'Operating_Profit','Operating Margin':'Operating_Margin','Sales Method':'Sales_Method'},inplace = True)
df


# In[7]:


# Replace characters in specific columns using regular expressions
columns_to_replace = ['Price_per_Unit', 'Total_Sales', 'Operating_Profit']
df[columns_to_replace] = df[columns_to_replace].replace({'\$': '', ' ': ''}, regex=True)
print(df)


# In[8]:


#convertting the Operating margin column into float from percentage
df['Operating_Margin']=df['Operating_Margin'].apply(lambda my_data:float(my_data.strip('%'))/100)
df


# In[9]:


#converting the float price per unit column to int
df['Price_per_Unit'] =df['Price_per_Unit'].apply(lambda my_data:int(round(float(my_data))))
df


# In[10]:


columns_with_commas = ['Units_Sold', 'Total_Sales', 'Operating_Profit']
for column in columns_with_commas:
    df[column] = df[column].apply(lambda x: str(x).replace(',', ''))
print(df)


# In[11]:


#removing the " ' " for data 
df['Product'] = df['Product'].str.replace("'",'')
df


# In[12]:


df.info()
df.describe()


# In[13]:


product_values_to_filter = ['Mens Apparel']
filtered_df = df[df['Product'].isin(product_values_to_filter)]
print(filtered_df)


# In[14]:


distinct_products = df['Product'].unique()
print(distinct_products)
print(len(distinct_products))


# In[15]:


#counting the number of cities
distinct_cities = df['City'].unique()
N0Cities= len(distinct_cities)
print("There are",N0Cities,"and they are",distinct_cities)


# In[16]:


#counting the retailers in the data
distinct_retailers = df['Retailer'].unique()
count_retailers = len(distinct_retailers)
print("There are",count_retailers,"and they are",distinct_retailers)


# In[17]:


# Group by 'City' and 'Product' columns and then calculate the count
product_counts = df.groupby(['City', 'Product']).size().reset_index(name='Count')

print(product_counts)


# In[18]:


import matplotlib.pyplot as plt

# Assuming 'product_counts' is the DataFrame with product counts by city
# Example: product_counts = df.groupby(['City', 'Product']).size().reset_index(name='Count')

# Plot a bar chart to visualize product counts by city
plt.figure(figsize=(10, 6))
plt.bar(product_counts['City'], product_counts['Count'])
plt.xlabel('City')
plt.ylabel('Product Count')
plt.title('Product Counts by City')
plt.xticks(rotation=90)
plt.show()


# In[19]:


Retailer_counts = df.groupby(['Retailer', 'Product']).size().reset_index(name='Count')

print(Retailer_counts)


# In[20]:


plt.figure(figsize=(10, 6))
plt.bar(Retailer_counts['Retailer'], Retailer_counts['Count'])
plt.xlabel('Retailer')
plt.ylabel('Retailer counts')
plt.title('Product Counts by Retailer')
plt.xticks(rotation=90)
plt.show()


# In[22]:


max_Operating_Profit = df['Operating_Profit'].max()
max_Operating_Profit


# In[23]:


min_Operating_Profit = df['Operating_Profit'].min()
min_Operating_Profit


# In[24]:


Operating_Profit= df.groupby(['Operating_Profit', 'Product']).size().reset_index(name='Count')
print(Operating_Profit)


# In[25]:


Total_Sales = df.groupby(['Product', 'Total_Sales']).size().reset_index(name='count')

print(Total_Sales)


# In[26]:


SalesMethod = df.groupby(['Sales_Method', 'Retailer']).size().reset_index(name='count')
print(SalesMethod)


# In[27]:


sales_method_counts = df['Sales_Method'].value_counts().reset_index()
sales_method_counts.columns = ['Sales_Method', 'Count']

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(sales_method_counts['Sales_Method'], sales_method_counts['Count'])
plt.xlabel('Sales Method')
plt.ylabel('Sales Count')
plt.title('Sales Count by Sales Method')
plt.xticks(rotation=45)
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt

retailer_counts = df['Retailer'].value_counts().reset_index()
retailer_counts.columns = ['Retailer', 'Count']

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(retailer_counts['Retailer'], retailer_counts['Count'])
plt.xlabel('Retailer')
plt.ylabel('Count')
plt.title('Count by Retailer')
plt.xticks(rotation=45)
plt.show()


# In[29]:


product_sales = df.groupby('Product')['Total_Sales'].sum().reset_index()

# Sort the results in descending order of sales
product_sales_sorted = product_sales.sort_values(by='Total_Sales', ascending=False)

# Get the most sold product (top product)
most_sold_product = product_sales_sorted.iloc[0]
print("Most Sold Product:", most_sold_product['Product'])


# In[31]:


product_sales_counts = df['Product'].value_counts()

print(product_sales_counts)


# In[54]:


# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'Product' and calculate the sum of 'Operating_Profit' for each product
product_profits = df.groupby('Product')['Operating_Profit'].sum().reset_index()

# Find the product with the highest total operating profit
most_profitable_product = product_profits.loc[product_profits['Operating_Profit'].idxmax()]
#/033[30m is the color code
print("\033[30m","Most Profitable Product:", "\033[32m",most_profitable_product['Product'])
print("\033[30m","Total Operating Profit:", "\033[32m",most_profitable_product['Operating_Profit'])


# In[33]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'Product' and calculate the sum of 'Operating_Profit' for each product
product_profits = df.groupby('Product')['Operating_Profit'].sum().reset_index()

# Find the product with the lowest total operating profit
least_profitable_product = product_profits.loc[product_profits['Operating_Profit'].idxmin()]

print("Least Profitable Product:", least_profitable_product['Product'])
print("Total Operating Profit:", least_profitable_product['Operating_Profit'])


# In[53]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'Retailer' and calculate the sum of 'Operating_Profit' for each retailer
retailer_profits = df.groupby('Retailer')['Operating_Profit'].sum().reset_index()

# Find the retailer company with the highest total operating profit
most_profitable_retailer = retailer_profits.loc[retailer_profits['Operating_Profit'].idxmax()]

print("\033[30m","Most Profitable Retailer:", "\033[32m",most_profitable_retailer['Retailer'])
print("\033[30m","Total Operating Profit:","\033[32m", most_profitable_retailer['Operating_Profit'])


# In[52]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'Retailer' and calculate the sum of 'Operating_Profit' for each retailer
retailer_profits = df.groupby('Retailer')['Operating_Profit'].sum().reset_index()

# Find the retailer company with the lowest total operating profit
least_profitable_retailer = retailer_profits.loc[retailer_profits['Operating_Profit'].idxmin()]

print("\033[30m","Least Profitable Retailer:", "\033[32m",least_profitable_retailer['Retailer'])
print("\033[30m","Total Operating Profit:","\033[32m", least_profitable_retailer['Operating_Profit'])


# In[56]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'City' and calculate the sum of 'Operating_Profit' for each city
city_profits = df.groupby('City')['Operating_Profit'].sum().reset_index()

# Find the city with the highest total operating profit
highest_profit_city = city_profits.loc[city_profits['Operating_Profit'].idxmax()]

print("\033[30m","City with Highest Profits:","\033[32m", highest_profit_city['City'])
print("\033[30m","Total Operating Profit:","\033[32m", highest_profit_city['Operating_Profit'])
# Find the city with the lowest total operating profit
lowest_profit_city = city_profits.loc[city_profits['Operating_Profit'].idxmin()]

print("\033[30m","City with Lowest Profits:","\033[31m", lowest_profit_city['City'])
print("\033[30m","Total Operating Profit:","\033[31m", lowest_profit_city['Operating_Profit'])


# In[57]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Extract year and month from the 'Date' column
df['Year'] = df['Invoice_Date'].dt.year
df['Month'] = df['Invoice_Date'].dt.month

# Group by year and month and calculate the sum of 'Operating_Profit' for each period
profit_by_period = df.groupby(['Year', 'Month'])['Operating_Profit'].sum().reset_index()

# Find the period with the highest and lowest total operating profit
highest_profit_period = profit_by_period.loc[profit_by_period['Operating_Profit'].idxmax()]
lowest_profit_period = profit_by_period.loc[profit_by_period['Operating_Profit'].idxmin()]

print("\033[30m","Highest Profits Period (Year-Month):","\033[32m", highest_profit_period['Year'], "-", highest_profit_period['Month'])
print("\033[30m","Total Operating Profit:","\033[32m", highest_profit_period['Operating_Profit'])

print("\033[30m","Lowest Profits Period (Year-Month):","\033[31m", lowest_profit_period['Year'], "-", lowest_profit_period['Month'])
print("\033[30m","Total Operating Profit:","\033[31m", lowest_profit_period['Operating_Profit'])


# In[58]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'Region' and calculate the sum of 'Operating_Profit' for each region
region_profits = df.groupby('Region')['Operating_Profit'].sum().reset_index()

# Find the region with the highest total operating profit
highest_profit_region = region_profits.loc[region_profits['Operating_Profit'].idxmax()]

print("\033[30m","Region with Highest Profits:","\033[32m", highest_profit_region['Region'])
print("\033[30m","Total Operating Profit:", "\033[32m",highest_profit_region['Operating_Profit'])

# Find the region with the lowest total operating profit
lowest_profit_region = region_profits.loc[region_profits['Operating_Profit'].idxmin()]

print("\033[30m","Region with Lowest Profits:", "\033[31m",lowest_profit_region['Region'])
print("\033[30m","Total Operating Profit:", "\033[31m",lowest_profit_region['Operating_Profit'])


# In[59]:


df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')

# Group by 'Product' and calculate the sum of 'Units_Sold' for each product
product_units_sold = df.groupby('Product')['Units_Sold'].sum().reset_index()

# Find the product with the highest total unit sales
highest_units_sold_product = product_units_sold.loc[product_units_sold['Units_Sold'].idxmax()]
lowest_units_sold_product = product_units_sold.loc[product_units_sold['Units_Sold'].idxmin()]

print("Product with Highest Unit Sales:", "\033[32m",highest_units_sold_product['Product'])
print("\033[30m","Total Units Sold:","\033[32m", highest_units_sold_product['Units_Sold'])
print("\033[30m","Product with Lowest Unit Sales:", "\033[31m",lowest_units_sold_product['Product'])
print("\033[30m","Total Units Sold:", "\033[31m",lowest_units_sold_product['Units_Sold'])


# In[60]:


df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')

# Group by 'Product' and calculate the sum of 'Units_Sold' for each product
product_units_sold = df.groupby('Retailer')['Units_Sold'].sum().reset_index()

# Find the product with the highest total unit sales
highest_units_sold_product = product_units_sold.loc[product_units_sold['Units_Sold'].idxmax()]
lowest_units_sold_product = product_units_sold.loc[product_units_sold['Units_Sold'].idxmin()]

print("\033[30m","Retailer with Highest Unit Sales:", "\033[32m",highest_units_sold_product['Retailer'])
print("\033[30m","Total Units Sold:","\033[32m", highest_units_sold_product['Units_Sold'])
print("\033[30m","Retailer with Lowest Unit Sales:", "\033[31m",lowest_units_sold_product['Retailer'])
print("\033[30m","Total Units Sold:","\033[31m", lowest_units_sold_product['Units_Sold'])


# In[62]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Extract year from the 'Invoice_Date' column
df['Year'] = df['Invoice_Date'].dt.year

# Group by year and sales method, and calculate the sum of 'Operating_Profit' for each combination
profit_by_year_method = df.groupby(['Year', 'Sales_Method'])['Operating_Profit'].sum().reset_index()

# Find the combination of year and sales method with the highest total operating profit
highest_profit_year_method = profit_by_year_method.loc[profit_by_year_method['Operating_Profit'].idxmax()]

print("\033[30m","Year with Highest Profit:", "\033[32m",highest_profit_year_method['Year'])
print("\033[30m","Sales Method with Highest Profit:","\033[32m", highest_profit_year_method['Sales_Method'])
print("\033[30m","Total Operating Profit:","\033[32m", highest_profit_year_method['Operating_Profit'])


# In[68]:


import pandas as pd

# Assuming 'df' is your DataFrame

# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Extract year from the 'Invoice_Date' column
df['Year'] = df['Invoice_Date'].dt.year

# Filter data for the year 2020
data_2020 = df[df['Year'] == 2020]

# Group by 'Retailer' and aggregate the sum of 'Unit_Solds' and 'Operating_Profit'
retailer_data = data_2020.groupby('Retailer').agg({'Units_Sold': 'sum', 'Operating_Profit': 'sum'}).reset_index()

print(retailer_data)


# In[69]:


import pandas as pd
# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Extract year from the 'Invoice_Date' column
df['Year'] = df['Invoice_Date'].dt.year

# Filter data for the year 2020
data_2020 = df[df['Year'] == 2020]

# Group by 'Retailer' and aggregate the sum of 'Unit_Solds' and 'Operating_Profit'
retailer_data = data_2020.groupby('Retailer').agg({'Units_Sold': 'sum', 'Operating_Profit': 'sum'}).reset_index()

# Sort the data by 'Operating_Profit' in descending order
sorted_retailer_data = retailer_data.sort_values(by='Operating_Profit', ascending=False)

print(sorted_retailer_data)


# In[71]:


import pandas as pd
# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Extract year from the 'Invoice_Date' column
df['Year'] = df['Invoice_Date'].dt.year

# Filter data for the year 2020 and 'Outlet' sales method
data_outlet = df[(df['Year'] == 2020) & (df['Sales_Method'] == 'Outlet')]

# Calculate the total sum of sales for 'Outlet' sales method
total_outlet_sales = data_outlet['Units_Sold'].sum()

print("Total Outlet Sales for 2020:", total_outlet_sales)


# In[72]:


import pandas as pd
# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'Outlet']

# Calculate the total sum of sales for 'Outlet' sales method
total_outlet_sales = data_outlet['Units_Sold'].sum()

print("Total Outlet Sales:", total_outlet_sales)


# In[74]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'Outlet']

# Calculate the total sum of sales for 'Outlet' sales method
total_outlet_sales = data_outlet['Units_Sold'].sum()

print("Total Outlet Sales:", total_outlet_sales)


# In[78]:


df['Total_Sales'] = pd.to_numeric(df['Total_Sales'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'Outlet']

# Calculate the total sum of sales for 'Outlet' sales method
total_outlet_sales = data_outlet['Units_Sold'].sum()

print("Total operating profit Outlet Sales:", total_outlet_sales)


# In[79]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'Online']

# Calculate the total sum of sales for 'Outlet' sales method
total_online_sales = data_outlet['Units_Sold'].sum()

print("Total operating profit Online Sales:", total_online_sales)


# In[80]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Convert 'Invoice_Date' column to datetime type if not already
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'In-store']

# Calculate the total sum of sales for 'Outlet' sales method
total_instore_sales = data_outlet['Units_Sold'].sum()

print("Total operating profit In-store Sales:", total_instore_sales)


# In[82]:


df['Operating_Margin'] = pd.to_numeric(df['Operating_Margin'], errors='coerce')

# Filter data for the 'Outlet' sales method
data_outlet = df[df['Sales_Method'] == 'In-store']

# Calculate the total sum of sales for 'Outlet' sales method
total_instore_sales = data_outlet['Units_Sold'].sum()

print("Total operating margin In-store Sales:", total_instore_sales)


# In[83]:


# Convert 'Operating_Profit' column to numeric data type
df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'State' and calculate the sum of 'Operating_Profit' for each state
state_operating_profit = df.groupby('State')['Operating_Profit'].sum().reset_index()

# Sort the data in descending order based on 'Operating_Profit'
sorted_state_operating_profit = state_operating_profit.sort_values(by='Operating_Profit', ascending=False)

print(sorted_state_operating_profit)


# In[84]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'State' and calculate the sum of 'Operating_Profit' for each state
state_operating_profit = df.groupby('Region')['Operating_Profit'].sum().reset_index()

# Sort the data in descending order based on 'Operating_Profit'
sorted_state_operating_profit = state_operating_profit.sort_values(by='Operating_Profit', ascending=False)

print(sorted_state_operating_profit)


# In[85]:


df['Operating_Profit'] = pd.to_numeric(df['Operating_Profit'], errors='coerce')

# Group by 'State' and calculate the sum of 'Operating_Profit' for each state
state_operating_profit = df.groupby('Retailer')['Operating_Profit'].sum().reset_index()

# Sort the data in descending order based on 'Operating_Profit'
sorted_state_operating_profit = state_operating_profit.sort_values(by='Operating_Profit', ascending=False)

print(sorted_state_operating_profit)

