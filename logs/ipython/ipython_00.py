# IPython log file

get_ipython().magic(u'pinfo %logstart')
get_ipython().magic(u'logstart logs/ipython/')
get_ipython().magic(u'logstart logs/ipython/ipython_00.py')
data = pd.read_csv('data/data.csv')
data.columns
print(_)
for c in data.columns:
    print(c)
    
data.Status
data.Status.describe()
data.Status.hist()
data.Id
data.Refreshing
data.Refreshing.describe()
data.Refreshing.hist()
get_ipython().magic(u'pinfo data.Refreshing.hist')
data.Refreshing.hist()
data.groupby('Refreshing').size()
data.groupby('Id').size()
data.groupby('Id').size().size()
data.groupby('Id').size()
data.groupby('Runner').size()
sum(_)
data.dtypes
data.groupby('Baker').size()
data.groupby('Counter').size()
data.groupby('Regulator').size()
data.groupby('Has Python').size()
data.groupby('Has Whiteboard').size()
data.groupby('Has Reached Balmers Peak').size()
data.groupby('DNE').size()
data.groupby('Type of Activity Id').size()
data.dtypes
for c in data.columns:
    print(c.groupby(c).size())
    
for c in data.columns:
    print(data.groupby(c).size())
    
data.columns
features = data.columns[2:]
features
features = data.columns[2:].values
features
data[features].isnull()
_.any()
data.isnull().any()
get_ipython().magic(u'pinfo data.drop')
data.dtypes
float64
data.dtypes[-1]
data.columns[data.dtypes == dtype('float64')]
np.dtype
data.columns[data.dtypes == np.dtype('float64')]
data['Type of GPU Id'] * 10
blarg = _
blarg = data.loc['Type of GPU Id'] * 10
blarg = data.loc[:, 'Type of GPU Id'] * 10
blarg
blarg.dtype = np.dtype('int64')
blarg
int(blarg)
blarg.dtype = int
blarg.dtypes
blarg.dtype
blarg.dtypes = int
data.loc[:, 'Type of GPU Id'] = int(10*data['Type of GPU Id'])
data.loc[:, 'Type of GPU Id'] = (10*data['Type of GPU Id']).astype(int)
data['Type of GPU Id']
data['Type of GPU Id'].hist()
get_ipython().magic(u'pinfo pd.read_csv')
data.columns
import fw_data
data = fw_data.get_data()
import fw_data
data = fw_data.get_data()
get_ipython().magic(u'run fw_data.py')
data = get_data()
data
X, y, X_unk = get_data()
X.shape()
X
y
X
X_unk
X.shape
X
