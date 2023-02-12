import pandas as pd
path = './'
df = pd.read_csv(path+'datasetSentences.txt',sep ='	')
print(df.head())
