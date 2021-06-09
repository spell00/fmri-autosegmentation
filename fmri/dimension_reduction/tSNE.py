from fmri.utils.utils import ellipse_data, create_missing_folders
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ordination2d(data_frame, ORD=PCA,images_folder_path="/home/simon/results/annleukemia/plots/",filenames="NoName", a=0.5):
    type_images_folder_path = images_folder_path + filenames + "/"
    create_missing_folders(type_images_folder_path)
    try:
        assert type(data_frame) == pd.core.frame.DataFrame
    except:
        print("The type of the data object in pca2d has to be pandas.core.frame.DataFrame. Returning without finishing (no PCA plot was produced)")
        return

    y = np.array(data_frame.columns, dtype=str)
    classes_list = np.unique(y)

    ord = ORD(n_components=2,verbose=1)
    principalComponents = ord.fit_transform(np.transpose(data_frame.values))
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(y)], axis=1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component tSNE', fontsize=20)
    colors = ['r', 'g', 'b']
    for target, color in zip(classes_list, colors):
        indicesToKeep = finalDf[0] == target
        data1 = finalDf.loc[indicesToKeep, 'principal component 1']
        data2 = finalDf.loc[indicesToKeep, 'principal component 2']
        ellipse_data(data1, data2, ax, color)

        ax.scatter(data1, data2, c=color, s=12)
    ax.legend(classes_list)
    ax.grid()



    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, classes_list)

    plt.tight_layout()
    fig.tight_layout()

    fig.savefig(images_folder_path + type_ord + filenames + ".png")
    plt.close(fig)
