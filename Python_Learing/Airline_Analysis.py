"""
Consumer Airfare Analysis (Markets under 750 miles)

Author: Ryan Moh
Goal:
- Analyze airfare trends over time (short-haul markets).
- Compare legacy vs low-cost carriers (LCCs).
- Identify market patterns, pricing outliers, and competition effects.

This project shows ability to:
- Clean messy real-world data
- Create visualizations that explain business insights
- Apply statistical testing (t-test, Spearman correlation, effect size)
- Answer business questions like "Do more carriers lower fares?"

"""

# --- Libraries ---
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import ttest_ind, spearmanr


# --- Data Ingestion & Cleaning ---
filepath = r""

df = pd.read_csv(filepath, header = 0)

# Standardize column names (SLR)
df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")

#Carriers Clarification
carrier_dict = {
    # Legacy carriers
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "AS": "Alaska Airlines",

    # LCCs
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "VX": "Virgin America",  
    "SY": "Sun Country Airlines"
}

df["car"] = df["car"].map(carrier_dict).fillna(df["car"])



# Feature Enginerring
df["market"] = df["city1"] + "_" + df["city2"]

# Finding patterns there is no pattern with specific cities or years being exclude.
nulls_by_city = df.groupby("city1").apply(lambda x: x.isnull().sum().sum())
nulls_by_city = nulls_by_city[nulls_by_city > 0]
my_dict = nulls_by_city.to_dict()
print(my_dict)

#df.drop(['geocoded_city1','geocoded_city2'],axis = 1, inplace= True)

df_first10 = df[df['year'] < 2015]
df_last10 = df[df["year"] >= 2015]


# Insights: Market Concentration 
print("Number of carriers before 2015 vs after 2015:")
print("Before 2015:", df[df['year'] < 2015]['car'].nunique())
print("After 2015 :", df_last10['car'].nunique())

#Market fare last 10 years
print("Distrbution of market fare over 10 years: \n") 
print(df_last10["mkt_fare"].describe())


# Insights: Passenger count by Carrier
car_passengers = df_last10.groupby("car")["carpax"].sum().sort_values(ascending=False).head(5)
print("\nPassenger counts by carrier:\n", car_passengers)

car_passengers.head(5).plot(kind="bar", figsize=(10,5))
plt.title("Top 10 Carriers by Passenger Count")
plt.ylabel("Passengers")
plt.show()


# Insights: Market Share distribution 
car_shares = df_last10.groupby("car")["carpaxshare"].mean().sort_values(ascending=False)
print("\nAverage market share by carrier:\n", car_shares.round(2))

car_shares.head(5).plot(kind="pie", autopct="%.1f%%", figsize=(7,7))
plt.title("Market Share of Top 5 Carriers")
plt.show()


# Insight 4: Fare Trends Over Time
df_last10["fare_range"] = df_last10["fareinc_max"] - df_last10["fareinc_min"]
spreads = df_last10.sort_values("fare_range", ascending=False)
print(f"\n Fare Ranges \n")
print(spreads[["car", "fareinc_min", "fareinc_max", "fare_range"]].head(10))
plt.figure(figsize=(8,5))
sns.histplot(df_last10["fare_range"], bins=30, kde=True)
plt.title("Fare Range Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df_last10["fare_inc_x3paxsh"])
plt.title("How many passengers are 3 times minimum fair?")
plt.show()


# --- Time Series Analysis + Outliers identification --- 


# Overall Fare Trends past 10 years.
avg_fares_by_year = (df_last10.groupby("year")["caravgfare"].mean().reset_index())
print(f"{avg_fares_by_year}")

sns.lineplot(x="year", y="caravgfare", data=avg_fares_by_year, marker="o")
plt.title("Average Fares Over Time (All Markets)")
plt.xlabel("Year")
plt.ylabel("Average Fare")
plt.grid(True)
plt.show()

#--- Outliers Research ---
Q1 = avg_fares_by_year["caravgfare"].quantile(0.25)
Q3 = avg_fares_by_year["caravgfare"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = avg_fares_by_year[
    (avg_fares_by_year["caravgfare"] < lower_bound) |
    (avg_fares_by_year["caravgfare"] > upper_bound)
]

print("\n Outlier years based on IQR: \n")
print(outliers)


#2015 Outlier 
df_2015 = df[df["year"] == 2015]
car_fares_2015 = (df_2015.groupby("car")["caravgfare"].mean().sort_values(ascending=False))
print("\n2015 Average Fare by Carrier:\n", car_fares_2015.head(5))


#2015 Outliers by City
market_fares_2015 = (df_2015.groupby(["city1", "city2"])["caravgfare"].mean().sort_values(ascending=False))
print("\n2015 Highest-Fare Markets:\n", market_fares_2015.head(5))


car_fares_all_years = (df.groupby(["year", "car"])["caravgfare"].mean().reset_index())

sns.lineplot(x="year", y="caravgfare", hue="car", data=car_fares_all_years[car_fares_all_years["car"].isin(car_fares_2015.head(3).index)])
plt.title("Trend of Top 2015 Carriers Over Time")
plt.show()


#Average Far over time to citiesk
focus_cities = ["Dallas/Fort Worth, TX", "Fayetteville, AR", "Chicago, IL", "New York City, NY (Metropolitan Area)"]

color = {
    "Dallas/Fort Worth, TX": "tab:purple",
    "Fayetteville, AR": "tab:orange",
    "Chicago, IL": "tab:red",  
    "New York City, NY (Metropolitan Area)": "tab:green"
}

avg_all = (df_last10.groupby("year")["caravgfare"].mean().reset_index())

avg_by_city = (df_last10[df_last10["city2"].isin(focus_cities)].groupby(["year","city2"])["caravgfare"].mean().reset_index())

sns.lineplot(data=avg_all, x="year", y="caravgfare", marker="o", label="All markets")
sns.lineplot(data=avg_by_city, x="year", y="caravgfare", hue="city2", marker="o", palette=color)
plt.title("Average Fares Over Time")
plt.ylabel("Fare")
plt.legend()
plt.show()


# Legacy vs LCC Pricing
low_cost = {"Southwest Airlines","Spirit Airlines", "Frontier Airlines", "JetBlue Airways"}

legacy = {"American Airlines", "Delta Air Lines", "United Airlines", "Alaska Airlines"}

def carrier_type(x):
    if x in low_cost: return "LCC"
    if x in legacy: return "Legacy"
    return "Other"

df_last10["carrier_type"] = df_last10["car"].map(carrier_type)


# Yearly averages by carrier type
trend = (df_last10.groupby(["year","carrier_type"])["caravgfare"].mean().reset_index())

plt.figure(figsize=(9,5))
sns.lineplot(data=trend, x="year", y="caravgfare", hue="carrier_type", marker = "o")
plt.title("Fares Over Time by Carrier Type")
plt.ylabel("Average fare")
plt.grid(True)
plt.show()

# --- Hypothesis testing --- 
#Null: Legacy and LLC have the same mean fares, Alternative: Legacy have higher average fare than LLC, One tail T-test
legacy_fares = df_last10.loc[df_last10["carrier_type"]=="Legacy", "caravgfare"]
lcc_fares = df_last10.loc[df_last10["carrier_type"]=="LCC", "caravgfare"]
t_stat, p_val_two_tailed = ttest_ind(legacy_fares, lcc_fares, equal_var=False)

# One-tailed p-value (Legacy > LCC)
p_val_one_tailed = p_val_two_tailed / 2 if t_stat > 0 else 1
print(f"T-statistic {t_stat:.3f}")
print(f"One-tailed p-value {p_val_one_tailed:.4f}")
#Business impact: Legacy carriers consistently charge higher fares

#Cohen's d Usage
mean_diff = legacy_fares.mean() - lcc_fares.mean()
pooled_std = np.sqrt(((legacy_fares.var(ddof=1) + lcc_fares.var(ddof=1)) / 2))
cohens_d = mean_diff / pooled_std

print(f"Cohen's d {cohens_d:.3f}")
#Business impact:  1.906 Cohen'd, Legacy airfare has been consistenly more expensive than LCC  . 

#Does having more carriers lower the average price fare? 
market_comp = (df_last10.groupby(["year","market"]).agg(avg_fare=("caravgfare", "mean"),n_carriers=("car", "nunique"),pax=("carpax", "sum")).reset_index())
 
sns.scatterplot(data=market_comp, x="n_carriers", y="avg_fare", alpha=0.4)
sns.regplot(data=market_comp, x="n_carriers", y="avg_fare", scatter=False)
plt.title("Number of Carriers vs Average Fare")
plt.show()

rho, p = spearmanr(market_comp["n_carriers"], market_comp["avg_fare"])
print(f"Values: ρ={rho:.3f}, p={p:.4f} (carriers vs avg fare)")
#Business impact: Rho = -0.603, there is a negative relationship, more carriers the lower the average price
