from pandas.core.frame import DataFrame
from scipy.io import loadmat 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

##
# getting familiar with hyperspectral data
##


# loading the data

'''
X - input: 3D
y - output: 2D
Returns: data and ground truth
'''
def load_data():
    X = loadmat('data/PaviaU.mat')['paviaU']
    y = loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    #print(X[0])
    return X, y


X, y = load_data()


# gettin the pixels and saving to csv file
def get_pixels(X, y):
    # extract pixels/reshape:
    # each pixel is a vector of lenght of the number of bands of HSI
    q = X.reshape(-1, X.shape[2])

    # print(q[0])
    '''
    print("q shape: ", q.shape)
    q shape:  (207400, 103)
    610x340
    '''
    # data frame
    df = pd.DataFrame(data = q)

    # dataframe with classes
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis = 1) 

    # column names
    df.columns = [f'band {i}' for i in range(1, X.shape[2] + 1)] + ['class']
    #print(df.head())

    # save to csv file
    df.to_csv('data_set.csv')
    return df


df = get_pixels(X, y)

#print(df.head())


'''
exploring the data
''' 
###
# Dimension reduction
###


# reducing to 3D using PCA
# way to reduce the dimensionality of feature space
# refs: 
# https://online.stat.psu.edu/stat505/book/export/html/670
# https://www.youtube.com/watch?v=FgakZw6K1QQ
# https://www.youtube.com/watch?v=HMOI_lkzW08
# https://towardsdatascience.com/visualising-the-classification-power-of-data-54f5273f640

pca = PCA(n_components=3)

# data.iloc[<row selection>, <column selection>] //remove
dt = pca.fit_transform(df.iloc[:, :-1].values)


# print(dt[0])

# concatenate the data and the classes
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)

# set column names
q.columns = [f'PC - {i}' for i in range(1, 4)] + ['class']

print(q.head(10))

# save to vsv file
q.to_csv('data_set_pca.csv', index=False)

# remove class 0
qq = q[q['class'] != 0]
 # check counts for classes
print(qq['class'].value_counts())


# labels
class_labels = {'1': 'Asphalt',
'2' :'Meadows',
'3'	:'Gravel',
'4'	:'Trees',
'5'	:'Painted metal sheets',
'6'	:'Bare Soil',
'7'	:'Bitumen',
'8'	:'Self Blocking Bricks',
'9'	:'Shadows'}

# add class labels column
qq['class_labels'] = qq['class'].apply(lambda x: class_labels[str(x)])

# check counts for classes
print(qq['class_labels'].value_counts())

# chck head
print(qq.head())

# visualisation of data

#get counts
counts = qq['class_labels'].value_counts()


# TBC..






























# img_data = loadmat('data/PaviaU.mat')
# def display_band():
#     # check data
#     #print(img_data)
    
#     # check for keys
#     data_keys = list(img_data.keys())

#     # data from index 3: 'paviaU'
#     data_In = (img_data[data_keys[3]])
#     # check data
#     print(data_In)
#     # check shape
#     print(data_In.shape)

#     # visualise sample image, band at index 0
#     plt.figure(figsize=(10, 10))
#     plt.imshow(data_In[:, :, 102])
#     plt.show()


#     # index = 0
#     # for i in data_In:
#     #     plt.figure(figsize=(10, 10))
#     #     plt.imshow(data_In[:,:,index], interpolation='nearest')
#     #     plt.show()
#     #     index += 1

# #print(data_keys)

# display_band()











