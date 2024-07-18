import pandas as pd
import chardet


# when you are confident that the file is encoded in the default encoding or a known encoding that you can specify directly
df= pd.read_csv('train.csv')


# when you are dealing with files of unknown or potentially non-standard encodings. 
# This is common when working with data from various sources where encoding consistency cannot be guaranteed.
with open('train.csv','rb') as f:
	encoding = chardet.detect(f.read())['encoding']

Df = pd.read_csv('train.csv',encoding=encoding)
