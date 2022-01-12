import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
#print(data)

dataFirstRow = data[0][:,0]
dataSecondRow = data[0][:,1]
anotherCenter = data[1]

#plt.scatter(dataFirstRow, dataSecondRow, anotherCenter, cmap='rainbow') #plot all the rows of the first column against rows of second column
#plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
#can adjust this value and see what happens :)
kmeans.fit(data[0])
kmeans.cluster_centers_

kmeans.labels_ #use this to find labels in the data
fig , (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize = (10,6)) #they share an axis
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:, 1], c = kmeans.labels_, cmap='rainbow')
#^plot original data, color it based off of what algorithm though they should look like

ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:, 1], c = kmeans.labels_, cmap ='rainbow')
#plot data, base color off of what acutal data values are
plt.show()