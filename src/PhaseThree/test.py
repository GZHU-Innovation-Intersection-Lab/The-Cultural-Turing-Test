import pandas as pd
use = ['condition','model','region','persona']
dtype = {'condition':'category','model':'category','region':'category','persona':'category'}
df = pd.read_csv('llm_vsm_items_long_condition.csv', usecols=use, dtype=dtype)
out = (df.drop_duplicates()
         .groupby(['condition','model','region'])['persona']
         .nunique()
         .reset_index()
         .rename(columns={'persona':'N_personas'}))
out.to_csv('persona_counts_by_condition_model_region.csv', index=False)
print(out.to_string(index=False))
