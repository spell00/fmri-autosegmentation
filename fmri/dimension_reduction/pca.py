from fmri.utils.utils import ellipse_data, create_missing_folders
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_colors():
    n = 6
    color = plt.cm.coolwarm(np.linspace(0.1, 0.9, n))  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
    return hexcolor
def ordination2d(data_frame, ORD=PCA,images_folder_path="/home/simon/results/annleukemia/plots/",filename="pca", a=0.5):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    type_images_folder_path = images_folder_path + filename + "/"
    create_missing_folders(type_images_folder_path)
    try:
        assert type(data_frame) == pd.core.frame.DataFrame
    except:
        print("The type of the data object in pca2d has to be pandas.core.frame.DataFrame. Returning without finishing (no PCA plot was produced)")
        return

    y = np.array(data_frame.columns, dtype=str)
    classes_list = np.unique(y)

    pca = ORD(n_components=2)

    data_frame.values[np.isnan(data_frame.values)] = 0
    principalComponents = pca.fit_transform(data_frame.values)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(y)], axis=1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    #colors = get_colors()
    for target in classes_list:
        indicesToKeep = finalDf[0] == target
        data1 = finalDf.loc[indicesToKeep, 'principal component 1']
        data2 = finalDf.loc[indicesToKeep, 'principal component 2']
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        ellipse_data(data1, data2, ax)

        ax.scatter(data1, data2, s=20, alpha=a, linewidths=0, edgecolors='none')
    ax.legend(classes_list)
    ax.grid()


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, classes_list)

    plt.tight_layout()
    fig.tight_layout()
    fig.savefig(type_images_folder_path + "PCA2d" + filename + ".png")
    plt.close(fig)
