from pandas.core.frame import DataFrame
from scipy.io import loadmat 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

##
# getting familiar with hyperspectral data
##


'''
loading the data
X - input: 3D
y - output: 2D
Returns: data and ground truth
'''
def load_data():
    X = loadmat('data/PaviaU.mat')['paviaU']
    y = loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    #print(X[0])
    return X, y

X, y = load_data()


'''
gettin the pixels, examining shapes, 
making data frame and saving to csv file
'''
def get_pixels(X, y):
    # extract pixels/reshape:
    # each pixel is a vector of length of the number of bands of HSI

    # examine befor reshape
    print('X0')
    print(X[0])
    print('X0 shape')
    print(X[0].shape, '\n')

    # reshape
    q = X.reshape(-1, X.shape[2])
    
    # examine after reshape 
    print('Q0')
    print(q[0])
    print('Q0 shape')
    print(q[0].shape, '\n')

    # X shape
    print('X-shape ', X.shape)
    # q shape
    print('q-shape ', q.shape, '\n')


    # plot vector at index q[0]
    plt.plot(q[0])
    plt.show()

    '''
    print('q shape: ', q.shape)
    q shape:  (207400, 103)
    610x340
    '''


    # data frame
    df = pd.DataFrame(data = q)

    # dataframe with classes
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis = 1)

    # column names
    df.columns = [f'band {i}' for i in range(1, X.shape[2] + 1)] + ['class']
    print(df.head())

    # save to csv file
    df.to_csv('data_set.csv')
    return df


df = get_pixels(X, y)

print('\n')
print("PCA", '\n')

'''
Dimensionality reduction using PCA,
further exploration of the data
and adding class labels
''' 

# reducing to 3D using PCA
# a way to reduce the dimensionality of feature space
# refs: 
# https://online.stat.psu.edu/stat505/book/export/html/670
# https://www.youtube.com/watch?v=FgakZw6K1QQ
# https://www.youtube.com/watch?v=HMOI_lkzW08
# https://towardsdatascience.com/visualising-the-classification-power-of-data-54f5273f640

# PCA
pca = PCA(n_components=3)

# data.iloc[<row selection>, <column selection>] //remove
dt = pca.fit_transform(df.iloc[:, :-1].values)


#print('dt 207399 ', dt[207399])

# concatenate the data and the classes
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)

# set column names
q.columns = [f'PC - {i}' for i in range(1, 4)] + ['class']

#print(q.head(), '\n')


# save to vsv file
q.to_csv('data_set_pca.csv', index=False)

# remove class 0
qq = q[q['class'] != 0]

# check counts for classes
print(qq['class'].value_counts(), '\n')


# labels
labels = {'1': 'Asphalt',
'2' :'Meadows',
'3'	:'Gravel',
'4'	:'Trees',
'5'	:'Painted metal sheets',
'6'	:'Bare Soil',
'7'	:'Bitumen',
'8'	:'Self Blocking Bricks',
'9'	:'Shadows'}

# add class labels column
qq['labels'] = qq['class'].apply(lambda x: labels[str(x)])

# check counts for classes
print(qq['labels'].value_counts(), '\n')

# chck head
print(qq.head(), '\n')

# visualisation of data

#get counts
counts = qq['labels'].value_counts()


# # get counts for classes
# class_counts = qq['class'].value_counts()

#print(class_counts)

# set the bar plot with labels
plt.bar(range(len(counts)), counts, tick_label=counts.index)
# set the title
plt.title('Class Distribution')
# show the plot
plt.show()

# bar plot unins sns
# counts and labels
sns.barplot(x=counts.index, y=counts, data=qq)
# set title
plt.title('Class Distribution')
# show the plot
plt.show()



# set the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# set the title
ax.set_title('3D Scatter Plot')
# set the labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# set the colors
ax.scatter(qq.iloc[:, 0], qq.iloc[:, 1], qq.iloc[:, 2], c=qq['class'])
# show the plot
plt.show()
























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

#     # visualise sample image, band at index 102
#     plt.figure(figsize=(8, 8))
#     plt.imshow(data_In[:, :, 102])
#     plt.show()


#     # index = 0
#     # for i in data_In:
#     #     plt.figure(figsize=(8, 8))
#     #     plt.imshow(data_In[:,:,index], interpolation='nearest')
#     #     plt.show()
#     #     index += 1

# #print(data_keys)

# display_band()







