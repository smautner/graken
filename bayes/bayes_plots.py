'''
What does this do?
'''

import matplotlib
matplotlib.use("module://matplotlib-sixel")
import matplotlib.pyplot as plt
import numpy as np
import umap
def plot(X,Y):
    print("#######################")
    X = np.array(X)
    plt.scatter(X[:,0],X[:,2],c=Y)
    plt.show()
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()
    plt.scatter(X[:,1],X[:,2],c=Y)
    plt.show()
    '''
    # 3d doesnt look nice
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0],X[:,2],X[:,1],c=Y)
    plt.show()
    '''
    XU = umap.UMAP().fit_transform(X)
    plt.scatter(XU[:,0],XU[:,1],c=Y)
    plt.show()
    XUY = umap.UMAP().fit_transform(X, y= Y)
    plt.scatter(XUY[:,0],XUY[:,1],c=Y)
    plt.show()

#plot(X1,Y1)


