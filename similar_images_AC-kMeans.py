import sys, os, shutil
import numpy as np
sys.path.append("src")
from autoencoders.AE import AE
from clustering.KNN import KNearestNeighbours
from utilities.image_utilities import ImageUtils
from utilities.sorting import find_topk_unique
from utilities.plot_utilities import PlotUtils
from utilities.plot_kmeans_clustering import PlotCluster
from keras.backend import tensorflow_backend as backend
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # ========================================
    # Set run settings
    # ========================================

    # Choose autoencoder model
    #model_name = "simpleAE"
    model_name = "convAE"
    process_and_save_images = False  # image preproc: resize images and save?
    train_autoencoder = False  # train from scratch?

    # ========================================
    # Automated pre-processing
    # ========================================
    ##   Set flatten properties   ###
    if model_name == "simpleAE":
        flatten_before_encode = True
        flatten_after_encode = False
    elif model_name == "convAE":
        flatten_before_encode = False
        flatten_after_encode = True
    else:
        raise Exception("Invalid model name which is not simpleAE/convAE")

    img_shape = (100, 100)  # force resize -> (ypixels, xpixels)
    ratio_train_test = 0.8
    seed = 100

    loss = "binary_crossentropy"
    optimizer = "adam"
    n_epochs = 3000
    batch_size = 256

    save_reconstruction_on_load_model = True

    #clustering
    N_cluster = 20 #clustering
    
    # ========================================
    # Generate expected file/folder paths and settings
    # ========================================
    # Assume project root directory to be directory of file
    project_root = os.path.dirname(__file__)
    print("Project root: {0}".format(project_root))

    # Query and answer folder
    query_dir = os.path.join(project_root, 'test!')  # テストしたいqueryのパス
    answer_dir = os.path.join(project_root, 'output!')  # 結果のパス

    # In database folder
    db_dir = os.path.join(project_root, 'input')  # 訓練データのパス
    db_inventory_dir = os.path.join(project_root, 'input_15')  # 呼び出す在庫データのパス
    img_train_raw_dir = os.path.join(db_dir)
    img_inventory_raw_dir = os.path.join(db_inventory_dir)
    img_train_dir = os.path.join(db_dir)
    img_inventory_dir = os.path.join(db_inventory_dir)

    # Run output
    models_dir = os.path.join('models')

    # Set info file
    info = {
        # Run settings
        "img_shape": img_shape,
        "flatten_before_encode": flatten_before_encode,
        "flatten_after_encode": flatten_after_encode,

        # Directories
        "query_dir": query_dir,
        "answer_dir": answer_dir,

        "img_train_raw_dir": img_train_raw_dir,
        "img_inventory_raw_dir": img_inventory_raw_dir,
        "img_train_dir": img_train_dir,#訓練データのパス
        "img_inventory_dir": img_inventory_dir,

        # Run output
        "models_dir": models_dir
    }

    # Initialize image utilities (and register encoder)
    IU = ImageUtils()
    IU.configure(info)

    # Initialize plot utilities
    PU = PlotUtils()
    PC = PlotCluster()

    # ========================================
    #
    # Pre-process save/load training and inventory images
    #
    # ========================================

    # Process and save
    if process_and_save_images:

        # Training images
        IU.raw2resized_load_save(raw_dir=img_train_raw_dir,
                                 processed_dir=img_train_dir,
                                 img_shape=img_shape)
        # Inventory images
        IU.raw2resized_load_save(raw_dir=img_inventory_raw_dir,
                                 processed_dir=img_inventory_dir,
                                 img_shape=img_shape)


    # ========================================
    #
    # Train autoencoder
    #
    # ========================================

    # Set up autoencoder base class
    MODEL = AE()

    MODEL.configure(model_name=model_name)

    if train_autoencoder:

        print("Training the autoencoder...")

        # Generate naming conventions
        dictfn = MODEL.generate_naming_conventions(model_name, models_dir)
        MODEL.start_report(dictfn)  # start report

        # Load training images to memory (resizes when necessary)
        x_data_all, all_filenames = \
            IU.raw2resizednorm_load(raw_dir=img_train_dir, img_shape=img_shape)
        print("\nAll data:")
        print(" x_data_all.shape = {0}\n".format(x_data_all.shape))

        # Split images to training and validation set
        x_data_train, x_data_test, index_train, index_test = \
            IU.split_train_test(x_data_all, ratio_train_test, seed)
        print("\nSplit data:")
        print("x_data_train.shape = {0}".format(x_data_train.shape))
        print("x_data_test.shape = {0}\n".format(x_data_test.shape))

        # Flatten data if necessary
        if flatten_before_encode:
            x_data_train = IU.flatten_img_data(x_data_train)
            x_data_test = IU.flatten_img_data(x_data_test)
            print("\nFlattened data:")
            print("x_data_train.shape = {0}".format(x_data_train.shape))
            print("x_data_test.shape = {0}\n".format(x_data_test.shape))

        # Set up architecture and compile model
        MODEL.set_arch(input_shape=x_data_train.shape[1:],
                       output_shape=x_data_train.shape[1:])
        MODEL.compile(loss=loss, optimizer=optimizer)
        MODEL.append_arch_report(dictfn)  # append to report

        # Train model
        MODEL.append_message_report(dictfn, "Start training")  # append to report
        MODEL.train(x_data_train, x_data_test,
                    n_epochs=n_epochs, batch_size=batch_size)
        MODEL.append_message_report(dictfn, "End training")  # append to report

        # Save model to file
        MODEL.save_model(dictfn)

        # Save reconstructions to file
        MODEL.plot_save_reconstruction(x_data_test, img_shape, dictfn, n_plot=10)

    else:

        # Generate naming conventions
        dictfn = MODEL.generate_naming_conventions(model_name, models_dir)

        # Load models
        MODEL.load_model(dictfn)

        # Compile model
        MODEL.compile(loss=loss, optimizer=optimizer)

        # Save reconstructions to file
        if save_reconstruction_on_load_model:
            x_data_all, all_filenames = \
                IU.raw2resizednorm_load(raw_dir=img_train_dir, img_shape=img_shape)
            if flatten_before_encode:
                x_data_all = IU.flatten_img_data(x_data_all)
            MODEL.plot_save_reconstruction(x_data_all, img_shape, dictfn, n_plot=10)

    # ========================================
    #
    # Perform clustering recommendation
    #
    # ========================================

    # Load inventory images to memory (resizes when necessary)
    x_data_inventory, inventory_filenames = \
        IU.raw2resizednorm_load(raw_dir=img_inventory_dir, img_shape=img_shape)
    print("\nx_data_inventory.shape = {0}\n".format(x_data_inventory.shape))

    # Explictly assign loaded encoder
    encoder = MODEL.encoder

    # Encode our data, then flatten to encoding dimensions
    # We switch names for simplicity: inventory -> train, query -> test
    print("Encoding data and flatten its encoding dimensions...")
    if flatten_before_encode:  # Flatten the data before encoder prediction
        x_data_inventory = IU.flatten_img_data(x_data_inventory)
    x_train_kNN = encoder.predict(x_data_inventory)

    if flatten_after_encode:  # Flatten the data after encoder prediction
        x_train_kNN = IU.flatten_img_data(x_train_kNN)

    print("\nx_train_kNN.shape = {0}\n".format(x_train_kNN.shape))

    #=======================================
    #k_means
    #========================================
    print("Reading query images from query folder: {0}".format(query_dir))
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(x_train_kNN)
    predict_label = kmeans.labels_

        # =============================================
        #
        # Output results
        #
        # =============================================
    print(predict_label)
    for i in range(N_cluster):
        result_filename = os.path.join(answer_dir, "result_" + str(i) + "_cluster" +str(N_cluster) + ".png")
        label_index = np.where(predict_label==i)[0]
        print('index=', i,'n_=', len(label_index))
        PC.plot_cluster_answer(x_inventory=x_data_inventory,
                             label_index=label_index,
                             filename=result_filename,
                                n = len(label_index),
                              img_shape = img_shape)
# Driver
if __name__ == "__main__":
    main()
    backend.clear_session()
