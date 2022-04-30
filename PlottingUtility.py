import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# Took reference from : https://www.datacamp.com/community/tutorials/introduction-t-sne
def fashion_scatter(x: np.ndarray, y: np.ndarray):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(y))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[y.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    texts = []

    # for i in range(num_classes):

    #     # Position of each label at median of data points.

    #     x_text, y_text = np.median(x[y == i, :], axis=0)
    #     txt = ax.text(x_text, y_text, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     texts.append(txt)

    return f, ax, sc, texts