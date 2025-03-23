#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd

# Replace this with the actual path to the CSV file
csv_file_path = "US Airline Flight Routes and Fares 1993-2024 2.csv"

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_file_path)


df.head()


# In[24]:


df


# In[25]:


# Define a function to categorize the year
def categorize_period(year):
    if 2018 <= year <= 2020:
        return "Pre-Pandemic"
    elif 2021 <= year <= 2022:
        return "During-Pandemic"
    elif 2023 <= year <= 2024:
        return "Post-Pandemic"
    else:
        return "Other"

# Apply the function to create a new column
df['period'] = df['Year'].apply(categorize_period)


# In[26]:


df


# In[27]:


# Apply the function to create a new column
df['period'] = df['Year'].apply(categorize_period)

# Keep only the defined periods (drop "Other")
df = df[df['period'] != 'Other']

# Sort the DataFrame by Year and Quarter
df = df.sort_values(by=['Year', 'quarter']).reset_index(drop=True)

df


# In[33]:


# Step 1: Create a lookup dictionary from rows where geocoded data exists
geo_lookup = df.dropna(subset=['Geocoded_City1', 'Geocoded_City2']) \
               .drop_duplicates(subset=['city1', 'city2']) \
               .set_index(['city1', 'city2'])[['Geocoded_City1', 'Geocoded_City2']]

# Step 2: Define a function to fill missing geocoded values using the lookup
def fill_geocodes(row):
    if pd.isna(row['Geocoded_City1']) or pd.isna(row['Geocoded_City2']):
        key = (row['city1'], row['city2'])
        if key in geo_lookup.index:
            if pd.isna(row['Geocoded_City1']):
                row['Geocoded_City1'] = geo_lookup.loc[key, 'Geocoded_City1']
            if pd.isna(row['Geocoded_City2']):
                row['Geocoded_City2'] = geo_lookup.loc[key, 'Geocoded_City2']
    return row

# Step 3: Apply the function to the whole DataFrame
df = df.apply(fill_geocodes, axis=1)

df


# In[39]:


# --- Step 1: Define what a valid geocode is (must include city + coordinates)
def is_valid_geocode(val):
    return isinstance(val, str) and "," in val and "(" in val and ")" in val and not val.strip().startswith("(")

# --- Step 2: Create lookup from only valid geocoded entries
valid_geo = df[
    df['Geocoded_City1'].apply(is_valid_geocode) &
    df['Geocoded_City2'].apply(is_valid_geocode)
]

# Build lookup dictionary based on (city1, city2) pairs
geo_lookup = valid_geo.drop_duplicates(subset=['city1', 'city2']) \
                      .set_index(['city1', 'city2'])[['Geocoded_City1', 'Geocoded_City2']]

# --- Step 3: Function to fill in both missing and incomplete geocodes
def fill_geocodes(row):
    key = (row['city1'], row['city2'])

    if key in geo_lookup.index:
        # Fill if missing or incomplete
        if not is_valid_geocode(row['Geocoded_City1']):
            row['Geocoded_City1'] = geo_lookup.loc[key, 'Geocoded_City1']
        if not is_valid_geocode(row['Geocoded_City2']):
            row['Geocoded_City2'] = geo_lookup.loc[key, 'Geocoded_City2']
    
    return row

# --- Step 4: Apply to entire DataFrame
df = df.apply(fill_geocodes, axis=1)

# --- Step 5 (Optional): Check remaining bad rows
bad_rows = df[
    df['Geocoded_City1'].apply(lambda x: not is_valid_geocode(x)) |
    df['Geocoded_City2'].apply(lambda x: not is_valid_geocode(x))
]
print(f"Remaining incomplete geocoded rows: {len(bad_rows)}")

df


# In[40]:


df = df.drop(columns=['tbl'])

df


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt

# If needed, load the data
# df = pd.read_csv("your_cleaned_file.csv")

# Count the number of rows per year
year_counts = df['Year'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(year_counts.index, year_counts.values)
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.title("Number of Flight Records per Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[43]:


df.to_csv("filled_geocoded_airline_data.csv", index=False)


# In[ ]:




