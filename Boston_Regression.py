#python ~/Documents/'more applications'/ElementsofProgrammingInterviews/ML/Regression/Boston_Regression.py

import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
from matplotlib import pyplot as plt

if __name__=='__main__':
	print('REgression with boston data')

	boston=load_boston()
	dboston=pd.DataFrame(data=boston['data'],columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT'])
	dboston['target']=boston['target']
	print(dboston.head(10))
	print('shape',dboston.shape)
	print('describe: ',dboston.describe())
	print('info: ',dboston.info())
	print('is null sum')
	print(dboston.isnull().sum())
	sns.pairplot(dboston)
	plt.show()


