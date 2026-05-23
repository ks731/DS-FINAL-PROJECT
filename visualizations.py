import pandas as pd
import numpy as np
import plotly.express as px

#Load data 
merged = pd.read_csv('merged.csv')
merged = merged.drop_duplicates(subset='SOC').reset_index(drop=True)
merged = merged.copy()
print(f"Rows after deduplication: {len(merged)}")

#State columns 
state_cols = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
              'Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota',
              'Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
              'New Jersey','New Mexico','New York','North Carolina','North Dakota',
              'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
              'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
              'Virginia','Washington','West Virginia','Wisconsin','Wyoming']

#Calculate weighted average automation risk per state 
state_risk = {}

for state in state_cols:
    employment = merged[state]
    total_workers = employment.sum()
    if total_workers > 0:
        weighted_risk = (merged['Probability'] * employment).sum() / total_workers
        state_risk[state] = round(weighted_risk, 4)

state_df = pd.DataFrame({
    'State': list(state_risk.keys()),
    'Weighted Risk': list(state_risk.values())
})

print("\nTop 5 highest risk states:")
print(state_df.nlargest(5, 'Weighted Risk').to_string(index=False))
print("\nTop 5 lowest risk states:")
print(state_df.nsmallest(5, 'Weighted Risk').to_string(index=False))

#Choropleth map 
fig = px.choropleth(
    state_df,
    locations='State',
    locationmode='USA-states',
    color='Weighted Risk',
    scope='usa',
    color_continuous_scale='RdYlBu_r',
    range_color=[state_df['Weighted Risk'].min(), state_df['Weighted Risk'].max()],
    title='Average Automation Risk by U.S. State<br><sup>Weighted by state employment share per occupation</sup>',
    labels={'Weighted Risk': 'Automation Risk'}
)

fig.update_layout(
    geo=dict(showlakes=True, lakecolor='lightblue'),
    coloraxis_colorbar=dict(title='Risk Score'),
    title_font_size=18,
    margin=dict(l=0, r=0, t=60, b=0)
)

#Save as interactive HTML
fig.write_html('us_risk_map.html')
print("\nSaved us_risk_map.html")

#Save as static PNG for report
fig.write_image('us_risk_map.png', width=1200, height=700, scale=2)
print("Saved us_risk_map.png")