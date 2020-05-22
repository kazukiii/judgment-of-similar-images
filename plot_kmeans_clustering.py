"""
 plot_kmeans_clustering.py (author: kazukiii )
"""
import matplotlib.pyplot as plt
import numpy as np

class PlotCluster(object):
    def plot_cluster_answer(self, x_inventory=None, label_index=None, filename=None, gray_scale=False, n=5, img_shape=None):
        plt.figure(figsize=(2*n, 4))
        for j, index in enumerate(label_index):
            ax = plt.subplot(2, n, j + 1)
            x_inventory_plot = x_inventory[index].reshape((-1, img_shape[0], img_shape[1], 3))
            plt.imshow(np.squeeze(x_inventory_plot))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
