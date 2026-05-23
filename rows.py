import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all files
frey = pd.read_csv('automation_data_by_state.csv', encoding='latin-1')
skills = pd.read_csv('Skills.txt', sep='\t', encoding='latin-1')
abilities = pd.read_csv('Abilities.txt', sep='\t', encoding='latin-1')
knowledge = pd.read_csv('Knowledge.txt', sep='\t', encoding='latin-1')

# Pivot ONET files from long to wide 
skills_wide = skills.pivot_table(index='O*NET-SOC Code', columns='Element Name', values='Data Value')
abilities_wide = abilities.pivot_table(index='O*NET-SOC Code', columns='Element Name', values='Data Value')
knowledge_wide = knowledge.pivot_table(index='O*NET-SOC Code', columns='Element Name', values='Data Value')

for df in [skills_wide, abilities_wide, knowledge_wide]:
    df.columns.name = None

#  Merge O*NET files together 
onet = skills_wide.join(abilities_wide, how='inner', lsuffix='_skill', rsuffix='_ability')
onet = onet.join(knowledge_wide, how='inner', rsuffix='_knowledge')
onet = onet.reset_index()
onet['SOC'] = onet['O*NET-SOC Code'].str[:7]

#  Final merge 
merged = pd.merge(frey, onet, on='SOC', how='inner')
print(f"Rows: {merged.shape[0]}")
print(f"Columns: {merged.shape[1]}")

#  EDA: Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(merged['Probability'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Automation Probability')
axes[0].set_xlabel('Probability (0 = safe, 1 = high risk)')
axes[0].set_ylabel('Number of Occupations')

axes[1].boxplot(merged['Probability'], vert=False)
axes[1].set_title('Boxplot of Automation Probability')
axes[1].set_xlabel('Probability')

plt.tight_layout()
plt.savefig('eda_target_distribution.png')
plt.close()

print(f"Mean:   {merged['Probability'].mean():.3f}")
print(f"Median: {merged['Probability'].median():.3f}")
print(f"Std:    {merged['Probability'].std():.3f}")

# EDA: Missing values 
state_cols = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
              'Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota',
              'Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
              'New Jersey','New Mexico','New York','North Carolina','North Dakota',
              'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
              'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
              'Virginia','Washington','West Virginia','Wisconsin','Wyoming']

id_cols = ['SOC', 'Occupation', 'O*NET-SOC Code']
drop_cols = state_cols + id_cols
feature_cols = [c for c in merged.columns if c not in drop_cols and c != 'Probability']

null_counts = merged[feature_cols].isnull().sum()
null_pct = (null_counts / len(merged) * 100).round(1)

print(f"\nTotal feature columns: {len(feature_cols)}")
print(f"Columns with NO nulls: {(null_counts == 0).sum()}")
print(f"Columns with >50% null: {(null_pct > 50).sum()}")
print(f"\nTop 10 most null columns:")
print(null_pct.sort_values(ascending=False).head(10))

#EDA: Sample feature stats 
print(merged[feature_cols[:5]].describe().round(2))

merged.to_csv('merged.csv', index=False)
print("Saved merged.csv")