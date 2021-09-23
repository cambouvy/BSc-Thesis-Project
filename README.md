# QLKNN

*A collection of tools to create QuaLiKiz Neural Networks.*

Read the [GitLab Pages](https://karel-van-de-plassche.gitlab.io/QLKNN-develop/) and [wiki](https://gitlab.com/Karel-van-de-Plassche/QLKNN-develop/-/wikis/home) for more information.

## BTR Camille
This work was done for the the Bachelor Thesis Research project  "Modelling Turbulent Transport For The Tokamak L-Mode Edge using Neural Networks". As this work was carried out in collaboration with DIFFER within the QLKNN project, the code for this work is a branch of the QLKNN-develop project.
The files added to the branch for this Bachelor Thesis Research are the following:
* qlknn/dataset/filter_archive/edgerun_eight.py: Pre-processes and filters the data
* qlknn/training/gridSearch.py: Grid search on the L2 and batch size parameters (this file was split in 25 files - one per NN - for the sake of parallelization in the folder grid_searches_hyperparam)
* qlknn/training/gridSearch_structure.py: Grid search on the number of hidden layers and nodes per layer parameters (this file was split in 25 files - one per NN - for the sake of parallelization in the folder grid_searches_structure)
* qlknn/models/evaluateNN.py: Evaluate the RMSE of the grid search NNs
* qlknn/models/plotNN.py: Plots predictions of the NNs on linearly spaced input 
* qlknn/plot_NNs.py: Plot all the networks from the grid search including the data points using the quickslicer
* camille-scripts: Scripts tackling the data processing portion of the project

Additionally, some other methode were added in already existing files:
* negative_filter_to_zero in qlknn/dataset/filtering.py: Clips the predicted negative fluxes to zero
* determine_At in qlknn/dataset/hypercube_to_pandas.py: Determine At when Ati != Ate
* train2 in qlknn/training/keras_models.py, train2 and train_NDNN_from_folder2 in qlknn/training/train_NDNN.py: Allows to save multiple nn.json file when training (useful for the grid search)

Due to their large size, the datasets are not included in this repository.
