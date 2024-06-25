import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
ev_file_path = r'C:\Users\Utente\OneDrive\Desktop\ASRI\WashingtonEV.csv'
demographics_file_path = r'C:\Users\Utente\OneDrive\Desktop\ASRI\Washington_DemographicsByCounty_sample.csv'

ev_df = pd.read_csv(ev_file_path)
demographics_df = pd.read_csv(demographics_file_path)

# Display the first few rows of each dataset
print(ev_df.head())
print(demographics_df.head())
print("---------------------------------------------------------------------------------")

# Ensure consistent formatting
ev_df['County'] = ev_df['County'].str.strip().str.title()
demographics_df['County'] = demographics_df['County'].str.strip().str.title()

# Rename columns in demographics_df for consistency
demographics_df.rename(columns={'population': 'Population', 'county': 'County'}, inplace=True)

# Convert Population to numeric
demographics_df['Population'] = demographics_df['Population'].str.replace(',', '').astype(float)

# Merge datasets on 'County'
merged_df = pd.merge(ev_df, demographics_df, on='County', how='left')

# Display the first few rows of the merged dataset
print(merged_df.head())
print("---------------------------------------------------------------------------------")

# Define thresholds for categorization
def categorize_population(population):
    if population > 500000:
        return 'Urban'
    elif 50000 < population <= 500000:
        return 'Suburban'
    else:
        return 'Rural'

# Apply categorization
merged_df['Area Type'] = merged_df['Population'].apply(categorize_population)

# Display the first few rows to check the categorization
print(merged_df[['County', 'Population', 'Area Type']].head())
print("---------------------------------------------------------------------------------")

# Check for missing values
missing_values = merged_df[['County', 'City', 'Electric Vehicle Type']].isnull().sum()
print(missing_values)
print("---------------------------------------------------------------------------------")

# Drop missing values for relevant columns
merged_df = merged_df[['County', 'City', 'Electric Vehicle Type', 'Area Type']].dropna()
missing_values = merged_df[['County', 'City']].isnull().sum()
print(missing_values)
print("---------------------------------------------------------------------------------")

# Count the number of BEVs and PHEVs in each city and county
vehicle_counts_city = merged_df.groupby(['City', 'Electric Vehicle Type']).size().unstack(fill_value=0)
vehicle_counts_county = merged_df.groupby(['County', 'Electric Vehicle Type']).size().unstack(fill_value=0)

print("Vehicle counts by city:")
print(vehicle_counts_city.head())
print("---------------------------------------------------------------------------------")

print("Vehicle counts by county:")
print(vehicle_counts_county.head())
print("---------------------------------------------------------------------------------")

# Aggregate counts by Area Type and Electric Vehicle Type
area_type_distribution = merged_df.groupby(['Area Type', 'Electric Vehicle Type']).size().unstack(fill_value=0)

print("Distribution by area type:")
print(area_type_distribution)
print("---------------------------------------------------------------------------------")

# Conduct Chi-Square Test
chi2, p, dof, ex = chi2_contingency(area_type_distribution)
print(f"Chi-Square Test:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", ex)
print("---------------------------------------------------------------------------------")

# Proportion Comparison
proportions = area_type_distribution.div(area_type_distribution.sum(axis=1), axis=0)
print("Proportions by area type:")
print(proportions)

# Visualization
proportions.plot(kind='bar', stacked=True)
plt.title('Proportion of BEVs and PHEVs by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Proportion')
plt.legend(title='Electric Vehicle Type', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small') 
plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------")
# Count the number of BEVs and PHEVs in each county
county_distribution = merged_df.groupby(['County', 'Electric Vehicle Type']).size().unstack(fill_value=0)

# Stacked Bar Chart by County
county_distribution.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Distribution of BEVs and PHEVs by County')
plt.xlabel('County')
plt.ylabel('Count')
plt.legend(title='Electric Vehicle Type')
plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------")
# Proportions of BEVs and PHEVs in each area type
urban_proportions = area_type_distribution.loc['Urban']
suburban_proportions = area_type_distribution.loc['Suburban']
rural_proportions = area_type_distribution.loc['Rural']

# Pie Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].pie(urban_proportions, labels=urban_proportions.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Urban Areas')

axes[1].pie(suburban_proportions, labels=suburban_proportions.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Suburban Areas')

axes[2].pie(rural_proportions, labels=rural_proportions.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Rural Areas')

plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------")
# Box Plot of Electric Vehicle Counts by Area Type
vehicle_counts_area = merged_df.groupby(['Area Type', 'Electric Vehicle Type']).size().reset_index(name='Count')
sns.boxplot(x='Area Type', y='Count', hue='Electric Vehicle Type', data=vehicle_counts_area)
plt.title('Distribution of Electric Vehicle Counts by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Count')
plt.legend(title='Electric Vehicle Type')
plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------")
# Create a heatmap for area type and vehicle type correlation
heatmap_data = pd.crosstab(merged_df['Area Type'], merged_df['Electric Vehicle Type'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of BEVs and PHEVs by Area Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Area Type')
plt.tight_layout()
plt.show()


