# Import required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = r"C:\Users\x-mile\Desktop\rm_cw2\Results_21Mar2022.csv"
raw_data = pd.read_csv(file_path, encoding='utf-8')

# Select relevant columns and clean data
selected_columns = ['diet_group', 'sex', 'age_group', 'mean_ghgs', 'mean_land', 
                   'mean_watscar', 'mean_ghgs_ch4', 'mean_watuse']
clean_data = raw_data[selected_columns].dropna()

# Normalize numerical features
metrics_to_scale = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_ghgs_ch4', 'mean_watuse']
scaler = StandardScaler()
clean_data[metrics_to_scale] = scaler.fit_transform(clean_data[metrics_to_scale])

# Create aggregated view by demographic groups
grouped_data = clean_data.groupby(['diet_group', 'age_group', 'sex'])[metrics_to_scale].mean().reset_index()
grouped_data.head()  # Quick check of aggregated data

# Encode diet categories numerically
grouped_data['diet_code'] = grouped_data['diet_group'].astype('category').cat.codes

# Create interactive parallel coordinates plot
impact_plot = px.parallel_coordinates(
    grouped_data,
    color="diet_code",
    dimensions=metrics_to_scale,
    color_continuous_scale=px.colors.diverging.Tealrose,
    labels={'diet_code': 'Diet Category'},
    title="Diet-Related Environmental Impact Analysis (Standardized Metrics)"
)

# Enhance plot readability
impact_plot.update_layout(
    coloraxis_colorbar=dict(
        title="Diet Type",
        tickvals=grouped_data['diet_code'].unique(),
        ticktext=grouped_data['diet_group'].unique()
    ),
    width=1000,
    height=600
)
impact_plot.show()

# Radar chart visualization (requires plotly.graph_objects)
import plotly.graph_objects as go

def create_radar_chart(dataframe, diet_selection):
    """Generate radar chart comparing age groups for specific diet type"""
    filtered_data = dataframe[dataframe['diet_group'] == diet_selection]
    metric_labels = [col.replace('_', ' ').title() for col in metrics_to_scale]
    
    chart = go.Figure()
    for age_bracket in filtered_data['age_group'].unique():
        age_data = filtered_data[filtered_data['age_group'] == age_bracket]
        chart.add_trace(go.Scatterpolar(
            r=age_data[metrics_to_scale].values.flatten().tolist(),
            theta=metric_labels,
            fill='toself',
            name=f"Age {age_bracket}"
        ))
    
    chart.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"Environmental Impact Profile: {diet_selection.capitalize()} Diet",
        width=800,
        height=600
    )
    return chart

# Visualize environmental impact for meat-based diets
create_radar_chart(grouped_data, 'meat').show()

# Statistical analysis section
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Compare diet groups using ANOVA and post-hoc tests
for metric in metrics_to_scale:
    diet_groups = [grouped_data[grouped_data['diet_group'] == diet][metric] 
                  for diet in grouped_data['diet_group'].unique()]
    
    f_value, p_value = stats.f_oneway(*diet_groups)
    print(f"Metric: {metric}\nF-statistic: {f_value:.2f}, p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant differences found. Running post-hoc analysis:")
        tukey_results = pairwise_tukeyhsd(
            endog=grouped_data[metric],
            groups=grouped_data['diet_group'],
            alpha=0.05
        )
        print(tukey_results.summary())
    else:
        print("No significant differences between diet groups\n")
