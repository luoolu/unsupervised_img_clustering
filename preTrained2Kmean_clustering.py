# -*- coding: utf-8 -*-
# @Time : 2020/3/18 上午10:29
# @Author : LuoLu
# @FileName: preTrained2Kmean_clustering.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import csv
# Misc
import os
# Function to get the first mode number when there are multiple mode numbers
from collections import Counter

import matplotlib.cm as cm
# Matplot
import matplotlib.pyplot as plt
import numpy as np
# from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
# from keras.applications.resnet50 import preprocess_input
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
# Kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def get_first_mode(a):
    c = Counter(a)
    mode_count = max(c.values())
    mode = {key for key, count in c.items() if count == mode_count}
    first_mode = next(x for x in a if x in mode)
    return first_mode


# Part 1 - Feature Extraction using CNN

# Define CNN model
# For feature extraction, include_top has to be False
# Otherwise, with the top 3 fully-connected layers, result is all grouped to 1000 defined categories
# model = VGG16(weights='imagenet', include_top=False)
model = DenseNet201(weights='imagenet', include_top=False)

img_path = 'result/'
dirs = os.listdir(img_path)

ResNet50_feature_list = []

# Pass all the images to CNN model
for file in dirs:
    # Image preprocessing
    # Input layer of CNN model only takes image in sie of 224 x 224.  Resize the image to fit the size.
    img = image.load_img(img_path + file, target_size=(224, 224), color_mode='rgb')
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    # Image feature
    ResNet50_feature = model.predict(img_arr)
    ResNet50_feature_arr = np.array(ResNet50_feature)
    ResNet50_feature_list.append(ResNet50_feature_arr.squeeze())

# print('ResNet50 predict Finish！！')
# Convert the feature list for 5011 images into array
ResNet50_feature_list_arr = np.array(ResNet50_feature_list)
# print("ResNet50_feature_list_arr shape: ", ResNet50_feature_list_arr.shape)
# img_r = img.reshape((shape[0], shape[1] * shape[2]))
ResNet50_feature_list_arr = ResNet50_feature_list_arr.reshape(ResNet50_feature_list_arr.shape[0],
                                                              ResNet50_feature_list_arr.shape[1] *
                                                              ResNet50_feature_list_arr.shape[2] *
                                                              ResNet50_feature_list_arr.shape[3])
print("ResNet50_feature_list_arr reshape shape: ", ResNet50_feature_list_arr.shape)

# Part 2 - Clustering using Kmeans

# range_n_clusters = list(range(2, 10))
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# range_n_clusters = [2,	3,	4,	5,	6,	7,	8,	9,	10,	20,	30,	40,	50,	60,	70,	80,	90,	100,	200,	300,	400,	500, 600,	700,	800,	900,	1000, 2000, 3000]
cluster_silhouette_score = []

# Try different clusters for comparison
for n_clusters in range_n_clusters:
    # Fit features into Kmeans
    # print('enter kmeans fit:')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ResNet50_feature_list_arr)
    # print('enter kmeans fit Finish!!')
    cluster_labels = kmeans.labels_
    # print('cluster_labels Finish!!')
    silhouette_avg = silhouette_score(ResNet50_feature_list_arr, cluster_labels)
    # print('silhouette_avg Finish!!')
    cluster_silhouette_score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Graph Plot - Code reference scikit-learn.org

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(ResNet50_feature_list_arr) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(ResNet50_feature_list_arr, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(ResNet50_feature_list_arr[:, 0], ResNet50_feature_list_arr[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=16, fontweight='bold')
    plt.savefig("ReadMeImg/" + str(n_clusters) + ".png")

# Part 3 - Pick the best number of cluster of choosing the largest Silhouette Score
best_n_cluster = range_n_clusters[cluster_silhouette_score.index(max(cluster_silhouette_score))]
print("Best number of clusters is =", best_n_cluster,
      "The best silhouette_score is :", max(cluster_silhouette_score))
kmeans = KMeans(n_clusters=best_n_cluster, random_state=0).fit(ResNet50_feature_list_arr)
cluster_labels = kmeans.labels_

# Part 4 - Reformat Output
clustered_images = [[] for x in range(len(set(cluster_labels)))]

for index in range(len(dirs)):
    clustered_images[cluster_labels[index]].append(dirs[index])

max_row = cluster_labels.tolist().count(get_first_mode(cluster_labels))
header = []
result = [[] for x in range(max_row)]

for col in range(max_row):
    for row in range(len(clustered_images)):
        if len(clustered_images[row]) <= col:
            result[col].append('')
        else:
            result[col].append("'" + clustered_images[row][col][:-4] + "'")
for row in range(len(clustered_images)):
    header.append('Cluster_%d' % row)
result.insert(0, header)

# plot result
plt.subplots(1, 1)
plt.title("Cluster Silhouette Score Visualization")
plt.xlabel('Range N Clusters')
plt.ylabel('Cluster Silhouette Score')
plt.plot(range_n_clusters, cluster_silhouette_score, marker='o', linewidth=3.0, color='red')
plt.grid(True)
plt.savefig("ReadMeImg/cluster_silhouette_score_DenseNet201.png")

with open('cluster_result.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(result)

csvFile.close()

