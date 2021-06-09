from fmri.utils.utils import ellipse_data, create_missing_folders
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
def get_colors():
    n = 6
    color = plt.cm.coolwarm(np.linspace(0.1, 0.9, n))  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
    return hexcolor


def ordination2d(data_frame, ord_type, images_folder_path, dataset_name, epoch, a=0.4, verbose=0, info="none",
                 show_images=True, df_valid=None, df_test=None, n=4):
    import pandas as pd
    import numpy as np

    pc1 = 'Component_1'
    pc2 = 'Component_2'

    type_images_folder_path = "/".join([images_folder_path, str(ord_type), str(dataset_name)]) + "/"
    type_images_folder_path = type_images_folder_path + info + "/"

    create_missing_folders(type_images_folder_path)


    try:
        assert type(data_frame) == pd.core.frame.DataFrame
    except:
        print("The type of the data object in pca2d has to be pandas.core.frame.DataFrame. Returning without finishing (no PCA plot was produced)")
        print(type(data_frame))
        exit()
        return
    if type(dataset_name) == list:
        names = [name for name in dataset_name]
        dataset_name = "_".join(names)

    y = np.array(data_frame.columns, dtype=str)
    classes_list = np.unique(y)
    data_frame.values[np.isnan(data_frame.values)] = 0
    ord = None
    ys = False
    if ord_type in ["pca", "PCA"]:
        ys = False
        ord = PCA(n_components=2)
    elif ord_type in ["kpca", "KPCA"]:
        ys = False
        ord = KernelPCA(n_components=2, kernel="rbf")
    elif ord_type in ["tsne", "tSNE", "TSNE", "t-sne", "T-SNE", "t-SNE"]:
        ys = False
        ord = TSNE(n_components=2, verbose=verbose)
    elif ord_type in ["lda", "LDA", "flda", "FLDA"]:
        ys = True
        ord = LDA(n_components=2)
    elif ord_type in ["qda", "QDA"]:
        ord = QDA()
        ys = True
    else:
        print(ord_type)
        exit("No ordination of that name is implemented. Exiting...")
    if ys:
        principal_components = ord.fit_transform(np.transpose(data_frame.values), y=y)
        if df_valid is not None:
            pcs_valid = ord.transform(df_valid.values)
            pcs_valid = pd.DataFrame(data=pcs_valid,  columns=['principal component 1', 'principal component 2'])
            y_valid = df_valid.columns
            pcs_valid = pd.concat([pcs_valid, pd.DataFrame(y_valid)], axis=1)

            pcs_test = ord.transform(df_test.values)
            pcs_test = pd.DataFrame(data=pcs_test,  columns=['principal component 1', 'principal component 2'])
            y_test = df_valid.columns


            pcs_test = ord.transform(pcs_test.values)
            pcs_test = pd.concat([pcs_test, pd.DataFrame(y_test)], axis=1)

    else:
        principal_components = ord.fit_transform(np.transpose(data_frame.values))

    if ord_type == "pca":
        ev = ord.explained_variance_ratio_
        means = ord.mean_
        if sum(means < 0):
            means = means - min(means)
        means_ratio = means / np.sum(np.sum(means, axis=0)) * 100
        coeff = np.transpose(ord.components_)
        order_importance = list(reversed(np.argsort(means)))
        coeff, means_ratio = coeff[order_importance], means_ratio[order_importance]

        factors = np.array(data_frame.index)[order_importance]
        x = list(range(len(factors)))
        plt.xlabel("Initial Features")
        plt.ylabel("% of varaince explained")
        plt.title("% of the variance is explained by the initial features (Total:" + str(np.round(np.sum(ev) * 100, 2)) + ")")
        plt.xticks([x[0]], [factors[0]], rotation=45, fontsize=8)
        plt.plot(means_ratio)
        plt.tight_layout()
        plt.savefig(type_images_folder_path + info + "_" + str(epoch) + "_var_exaplined_2D.png", dpi=100)
        print("plot at ", type_images_folder_path)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, pd.DataFrame(y)], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    if ord_type not in "kpca":
        ev = ord.explained_variance_ratio_
        if len(ev) > 1:
            pc1 = pc1 + ': ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
            pc2 = pc2 + ': ' + str(np.round(ev[1] * 100, decimals=2)) + "%"

    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title('2 component Ordination', fontsize=20)

    # colors = cm.viridis(np.linspace(0, 1, len(classes_list)))
    colors = ["g", "b", "k", "r"]
    print("coeff shape", coeff.shape)
    if len(coeff) < n:
        n = len(coeff)

    for t, target in enumerate(classes_list):
        indices_to_keep = final_df[0] == target
        indices_to_keep = list(indices_to_keep)
        data1 = final_df.loc[indices_to_keep, 'principal component 1']
        data2 = final_df.loc[indices_to_keep, 'principal component 2']
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        ellipse_data(data1, data2, ax, colors[t])

        ax.scatter(data1, data2, s=10, alpha=a, c=colors[t])

        labels = factors
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1) + str(np.round(means_ratio[i], 2)),
                         color='g', ha='center', va='center')
            else:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(labels[i]) + str(np.round(means_ratio[i], 2)),
                         color='g',
                         ha='center', va='center')

    ax.legend(classes_list)
    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, classes_list)

    if df_valid is not None:
        for t, target in enumerate(classes_list):
            indices_to_keep = final_df[0] == target
            indices_to_keep = list(indices_to_keep)
            data1 = pcs_valid.loc[indices_to_keep, 'principal component 1']
            data2 = pcs_valid.loc[indices_to_keep, 'principal component 2']
            try:
                assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
            except:
                print("Nans were detected. Please verify the DataFrame...")
                exit()
            ellipse_data(data1, data2, ax, colors[t])

            ax.scatter(data1, data2, s=10, alpha=a)
        ax.legend(classes_list)
        ax.grid()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, classes_list)

    if df_test is not None:
        for t, target in enumerate(classes_list):
            indices_to_keep = final_df[0] == target
            indices_to_keep = list(indices_to_keep)
            data1 = pcs_test.loc[indices_to_keep, 'principal component 1']
            data2 = pcs_test.loc[indices_to_keep, 'principal component 2']
            try:
                assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
            except:
                print("Nans were detected. Please verify the DataFrame...")
                exit()
            ellipse_data(data1, data2, ax, colors[t])

            ax.scatter(data1, data2, s=10, alpha=a)
        ax.legend(classes_list)
        ax.grid()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, classes_list)




    try:
        plt.tight_layout()
        fig.tight_layout()
    except:
        pass
    plt.savefig(type_images_folder_path + info + "_" + str(epoch) + ".png", dpi=100)
    if show_images:
        plt.show()
    plt.close(fig)


