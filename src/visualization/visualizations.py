import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import folium
from folium.plugins import MarkerCluster

df = pd.read_pickle("../../data/processed/processed_data.pkl")

# Perform one-way ANOVA to test if there are significant differences in price across regions
anova_results = stats.f_oneway(
    *(df[df["Region"] == region]["Price"] for region in df["Region"].unique())
)

print(
    f"ANOVA F-statistic: {anova_results.statistic:.2f}, p-value: {anova_results.pvalue:.2e}"
)

# Check if the results are significant
if anova_results.pvalue < 0.05:
    print("There is a significant difference in property prices across regions.")
else:
    print("There is no significant difference in property prices across regions.")

# Perform Tukey's HSD test
tukey_results = pairwise_tukeyhsd(endog=df["Price"], groups=df["Region"], alpha=0.05)

# Print summary of Tukey's test results
print(tukey_results.summary())

# Plot Tukey's test results
tukey_results.plot_simultaneous(xlabel="Price", ylabel="Region")
plt.show()
plt.savefig("../../reports/figures/Tukey's Test Results.png")

# Assuming df is the DataFrame you're working with
# First, let's make a copy of the DataFrame to preserve the original data
df_log_transformed = df.copy()

# Apply a log transformation to the 'Price' column to reduce skewness
df_log_transformed["Log_Price"] = np.log1p(df_log_transformed["Price"])

# Now, let's create a boxplot with the log-transformed prices
plt.figure(figsize=(12, 6))
sns.boxplot(x="Region", y="Log_Price", data=df_log_transformed)
plt.title("Log-Transformed Price Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Log-Transformed Price (log($))")
plt.savefig("../../reports/figures/Log-Transformed Price Distribution by Region.png")
plt.show()

# Price Distribution in Each Region
g = sns.FacetGrid(df, col="Region", col_wrap=2, height=4)
g.map(
    sns.histplot,
    "Price",
    kde=True,
    color="purple",
    line_kws={"linewidth": 2, "color": "orange"},
)
g.set_titles("{col_name} Region")
g.set_axis_labels("Price", "Frequency")
plt.savefig("../../reports/figures/Price Distribution in Each Region.png")
plt.show()

# Scatter plot for Area vs. Price with different colors for each region
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Area", y="Price", data=df, hue="Region", palette="bright", alpha=0.6)
plt.title("Area vs. Price by Region")
plt.xlabel("Area (Square Feet)")
plt.ylabel("Price ($)")
plt.legend(title="Region")
plt.savefig("../../reports/figures/Area vs. Price.png")
plt.show()

# Scatter plot for Bedrooms vs. Price with different colors for each region
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x="Bedroom", y="Price", data=df, hue="Region", palette="bright", alpha=0.6
)
plt.title("Number of Bedrooms vs. Price by Region")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price ($)")
plt.legend(title="Region")
plt.savefig("../../reports/figures/Bedrooms vs. Price.png")
plt.show()

# Scatter plot for Bathrooms vs. Price with different colors for each region
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x="Bathroom", y="Price", data=df, hue="Region", palette="bright", alpha=0.6
)
plt.title("Number of Bathrooms vs. Price by Region")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price ($)")
plt.legend(title="Region")
plt.savefig("../../reports/figures/Bathrooms vs. Price.png")
plt.show()

# Correlation heatmaps for each region
for region in df["Region"].unique():
    regional_df = df[df["Region"] == region]
    numeric_columns = regional_df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        regional_df[numeric_columns].corr(), annot=True, cmap="coolwarm", linewidths=0.5
    )
    plt.title(f"Correlation Matrix for {region}")
    plt.savefig(f"../../reports/figures/Correlation heatmap {region}.png")
    plt.show()

# Geographical Distribution of Properties with Corrected Color Legend
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    x="Longitude",
    y="Latitude",
    data=df,
    hue="Price",
    size="Price",
    sizes=(20, 200),
    alpha=0.6,
    palette="viridis",
)
plt.title("Geographical Distribution of Properties")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Price ($)", loc="upper right")
plt.savefig("../../reports/figures/Geographical Distribution of Properties.png")
plt.show()

# Create HTML file for interactive map
# Create a base map
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

# Add a marker cluster to the map
marker_cluster = MarkerCluster().add_to(m)

# Add points to the map
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        fill=True,
        fill_color="blue"
        if row["Price"] <= 1500000
        else "green"
        if row["Price"] <= 3000000
        else "orange"
        if row["Price"] <= 4500000
        else "red",
        color=None,
        fill_opacity=0.7,
        popup=f"Price: ${row['Price']:,}",
    ).add_to(marker_cluster)

# Display the map
m.save(
    "../../interactive_map.html"
)  # This will save the interactive map as an HTML file
