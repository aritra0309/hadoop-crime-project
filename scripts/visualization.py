from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import folium
from folium import plugins
import json

spark = SparkSession.builder.appName("Crime Visualization").getOrCreate()

print("Loading aggregated data...")
state_agg = spark.read.parquet("output/state_agg_with_predictions")

# Convert to pandas for easier manipulation
state_df = state_agg.toPandas()

print("State data loaded:")
print(state_df.head(10))

# Indian states and union territories coordinates for mapping
state_coords = {
    "ANDHRA PRADESH": (15.9129, 78.6675),
    "ARUNACHAL PRADESH": (28.2180, 94.7278),
    "ASSAM": (26.2006, 92.9376),
    "BIHAR": (25.0961, 85.3131),
    "CHHATTISGARH": (21.4734, 81.6866),
    "GOA": (15.3017, 73.8207),
    "GUJARAT": (22.2587, 71.1924),
    "HARYANA": (29.0588, 77.0745),
    "HIMACHAL PRADESH": (31.1048, 77.1734),
    "JHARKHAND": (23.6102, 85.2799),
    "KARNATAKA": (15.3173, 75.7139),
    "KERALA": (10.8505, 76.2711),
    "MADHYA PRADESH": (22.9375, 78.6553),
    "MAHARASHTRA": (19.7515, 75.7139),
    "MANIPUR": (25.1458, 94.8670),
    "MEGHALAYA": (25.4670, 91.3662),
    "MIZORAM": (23.1815, 92.9789),
    "NAGALAND": (26.1584, 94.5624),
    "ODISHA": (20.9517, 85.0985),
    "PUNJAB": (31.1471, 75.3412),
    "RAJASTHAN": (27.0238, 74.2179),
    "SIKKIM": (27.5330, 88.5122),
    "TAMIL NADU": (11.1271, 79.2787),
    "TELANGANA": (18.1124, 79.0193),
    "TRIPURA": (23.4408, 91.9882),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.0668, 79.0193),
    "WEST BENGAL": (24.6551, 88.2038),
    "ANDAMAN AND NICOBAR": (12.2381, 92.7365),
    "CHANDIGARH": (30.7333, 76.8277),
    "DADRA AND NAGAR HAVELI": (20.1809, 73.0236),
    "DAMAN AND DIU": (20.6667, 72.8333),
    "LAKSHADWEEP": (12.2225, 73.1938),
    "PUDUCHERRY": (12.0657, 79.8711),
    "DELHI": (28.7041, 77.1025)
}

print("\n" + "="*50)
print("GENERATING CRIME HOTSPOTS MAP")
print("="*50)

# Create base map centered on India
india_center = [20.5937, 78.9629]
crime_map = folium.Map(location=india_center, zoom_start=5, tiles="OpenStreetMap")

# Aggregate crime by state across all years
state_crime = state_df.groupby("state")["total_crimes"].sum().reset_index()
state_crime = state_crime.sort_values("total_crimes", ascending=False)

print("\nTop states by total crimes:")
print(state_crime.head(10))

# Normalize crime values for color intensity
max_crimes = state_crime["total_crimes"].max()
min_crimes = state_crime["total_crimes"].min()

# Add markers for each state
for idx, row in state_crime.iterrows():
    state_name = row["state"].upper().strip()
    crimes = row["total_crimes"]
    
    if state_name in state_coords:
        lat, lon = state_coords[state_name]
        
        # Calculate color intensity (red gradient)
        intensity = (crimes - min_crimes) / (max_crimes - min_crimes)
        color = f"hsl(0, 100%, {100 - intensity*50}%)"  # Red shades
        
        # Add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + intensity * 15,
            popup=f"{state_name}<br>Total Crimes: {int(crimes)}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(crime_map)

# Add layer control
folium.LayerControl().add_to(crime_map)

# Save map
map_path = "output/crime_hotspots_map.html"
crime_map.save(map_path)
print(f"✓ Map saved to {map_path}")

# Generate summary statistics
print("\n" + "="*50)
print("GENERATING SUMMARY STATISTICS")
print("="*50)

summary_stats = []
summary_stats.append("=" * 70)
summary_stats.append("INDIA CRIME DATA ANALYSIS - SUMMARY REPORT")
summary_stats.append("=" * 70)
summary_stats.append("")

summary_stats.append("DATA PERIOD: 2001-2014")
summary_stats.append("")

summary_stats.append("OVERALL STATISTICS:")
total_crimes = state_df["total_crimes"].sum()
avg_crimes = state_df["total_crimes"].mean()
max_crimes_overall = state_df["total_crimes"].max()
min_crimes_overall = state_df["total_crimes"].min()

summary_stats.append(f"  Total Crimes (2001-2014): {int(total_crimes):,}")
summary_stats.append(f"  Average Crimes per Record: {int(avg_crimes):,}")
summary_stats.append(f"  Maximum Crimes in Single Year-State: {int(max_crimes_overall):,}")
summary_stats.append(f"  Minimum Crimes in Single Year-State: {int(min_crimes_overall):,}")
summary_stats.append("")

summary_stats.append("TOP 10 STATES BY TOTAL CRIMES:")
for rank, (idx, row) in enumerate(state_crime.head(10).iterrows(), 1):
    summary_stats.append(f"  {rank}. {row['state']}: {int(row['total_crimes']):,}")

summary_stats.append("")
summary_stats.append("YEARLY TRENDS:")
year_agg = state_df.groupby("year")["total_crimes"].sum().reset_index()
year_agg = year_agg.sort_values("year")
for _, row in year_agg.iterrows():
    summary_stats.append(f"  Year {int(row['year'])}: {int(row['total_crimes']):,}")

summary_stats.append("")
summary_stats.append("ANALYSIS OUTPUTS:")
summary_stats.append("  - data_preparation.py: Data loading, normalization, cleaning")
summary_stats.append("  - analytics.py: K-means clustering (5 clusters), Linear regression predictions")
summary_stats.append("  - visualization.py: Interactive Folium heatmap, summary statistics")

summary_stats.append("")
summary_stats.append("=" * 70)

# Write to file
summary_text = "\n".join(summary_stats)
with open("output/crime_analysis_summary.txt", "w") as f:
    f.write(summary_text)

print(summary_text)
print(f"\n✓ Summary saved to output/crime_analysis_summary.txt")

spark.stop()
print("\n✓ Visualization complete!")
