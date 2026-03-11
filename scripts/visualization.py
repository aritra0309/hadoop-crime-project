from pyspark.sql import SparkSession
import pandas as pd
import folium
import json
import os

# =====================================
# PATH SETUP
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# START SPARK
# =====================================

spark = SparkSession.builder.appName("Crime Visualization").getOrCreate()

time_series = spark.read.parquet(
    "hdfs://localhost:9000/crime/output/state_crime_time_series"
)

ts_df = time_series.toPandas()
spark.stop()

# =====================================
# CLEAN DATA
# =====================================

ts_df["state"] = ts_df["state"].str.strip().str.upper()

years = sorted(ts_df["year"].unique())

# =====================================
# LOAD GEOJSON
# =====================================

geojson_path = os.path.join(DATA_DIR, "india_states.geojson")

with open(geojson_path) as f:
    india_geojson = json.load(f)

# Normalize geojson names
for f in india_geojson["features"]:
    f["properties"]["NAME_1"] = f["properties"]["NAME_1"].strip().upper()

geo_states = [f["properties"]["NAME_1"] for f in india_geojson["features"]]

data_states = ts_df["state"].unique()

# =====================================
# STATE NAME MATCHING
# =====================================

state_match = {}

for g in geo_states:
    for d in data_states:
        if g in d or d in g:
            state_match[g] = d

# =====================================
# BUILD YEAR DATA USING GEOJSON NAMES
# =====================================

year_data = {}

for year in years:

    year_df = ts_df[ts_df["year"] == year]

    values = year_df.groupby("state")["total_crimes"].sum().to_dict()

    mapped = {}

    for geo_state in geo_states:

        if geo_state in state_match:

            mapped[geo_state] = values.get(state_match[geo_state], 0)

        else:

            mapped[geo_state] = 0

    year_data[str(year)] = mapped

# =====================================
# BUILD STATE TREND DATA
# =====================================

state_series = {}

for state in data_states:

    s = ts_df[ts_df["state"] == state].sort_values("year")

    state_series[state] = {
        "years": s["year"].tolist(),
        "values": s["total_crimes"].tolist()
    }

# =========================================================
# HEATMAP
# =========================================================

heat_map = folium.Map(
    location=[22.5,80],
    zoom_start=5,
    tiles="cartodbpositron"
)

script = f"""
<script>

var geojson = {json.dumps(india_geojson)};
var yearData = {json.dumps(year_data)};
var layer;

// Wait for Leaflet map to be ready
function waitForMap() {{
    if (typeof {heat_map.get_name()} === 'undefined') {{
        setTimeout(waitForMap, 100);
        return;
    }}
    var map = {heat_map.get_name()};

    function getColor(v) {{
        if (v > 500000) return "#800026";
        if (v > 350000) return "#BD0026";
        if (v > 250000) return "#E31A1C";
        if (v > 150000) return "#FC4E2A";
        if (v > 50000)  return "#FD8D3C";
        return "#FEB24C";
    }}

    function drawMap(year) {{
        if (layer) map.removeLayer(layer);
        layer = L.geoJson(geojson, {{
            style: function(feature) {{
                var state = feature.properties.NAME_1;
                var value = (yearData[year] && yearData[year][state]) ? yearData[year][state] : 0;
                return {{
                    fillColor: getColor(value),
                    weight: 1,
                    color: "black",
                    fillOpacity: 0.7
                }};
            }},
            onEachFeature: function(feature, layer) {{
                var state = feature.properties.NAME_1;
                var value = (yearData[year] && yearData[year][state]) ? yearData[year][state] : 0;
                layer.bindTooltip(state + ": " + value.toLocaleString() + " crimes");
            }}
        }}).addTo(map);
    }}

    window.drawMap = drawMap;
    drawMap("{years[0]}");
}}

waitForMap();

function updateYear() {{
    var y = document.getElementById("yearSelect").value;
    window.drawMap(y);
}}

</script>

<div style="
position:fixed;
top:10px;
left:50px;
z-index:9999;
background:white;
padding:10px;
border:2px solid grey">
<b>Select Year</b><br>
<select id="yearSelect" onchange="updateYear()">
{''.join([f'<option value="{y}">{y}</option>' for y in years])}
</select>
</div>
"""
heat_map.get_root().html.add_child(folium.Element(script))

heat_output = os.path.join(OUTPUT_DIR,"crime_heatmap_year.html")

heat_map.save(heat_output)

# =========================================================
# STATE TREND CHART
# =========================================================

chart_html = f"""
<html>

<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>

<div style="width:900px;margin:auto">

<h2>Crime Trend by State</h2>

<select id="stateSelect" onchange="updateChart()">
{''.join([f'<option value="{s}">{s}</option>' for s in data_states])}
</select>

<canvas id="chart"></canvas>

</div>

<script>

var stateSeries={json.dumps(state_series)};

var ctx=document.getElementById('chart').getContext('2d');

var chart=new Chart(ctx,{{
type:'line',
data:{{
labels:[],
datasets:[{{
label:'Crimes',
data:[],
borderWidth:3
}}]
}}
}});

function updateChart(){{

var s=document.getElementById("stateSelect").value;

var d=stateSeries[s];

chart.data.labels=d.years;
chart.data.datasets[0].data=d.values;

chart.update();

}}

updateChart();

</script>

</body>
</html>
"""

chart_output = os.path.join(OUTPUT_DIR,"state_trend_chart.html")

with open(chart_output,"w") as f:
    f.write(chart_html)

print("✓ Heatmap saved:",heat_output)
print("✓ Trend chart saved:",chart_output)