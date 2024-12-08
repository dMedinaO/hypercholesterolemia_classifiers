## Design and validation of a predictive protocol for early determination of maternal supraphysiological hypercholesterolemia in pregnancy

This repository facilitates the implementation and repliction of trained models implemented in the work: *Design and validation of a predictive protocol for early determination of maternal supraphysiological hypercholesterolemia in pregnancy*.

### General description of the repository

This repository has the following folders and files:

- **raw_data**: This folder has the raw dataset in excel format
- **src**: This folder contains the source code used during the project, including the scripts to process the dataset, the development of statistical analysis, training predictive models, and explainable strategies.
- **results**: This folder has all results and figures generated during the project, including the processed dataset, the descriptive process, the trained models, and the explainability results.

Moreover, the following files are including:

- *environment.yml*: facilitates the creation of the environment with all packages. Please, use the following command to create the environment based on the required packages:

```
conda env create -f environment.yml
```

- *LICENSE*: File with the license associated with this project.

### Strategies applied to train predictive models

The steps associated to the work implies:

1. Preprocessing dataset

2. Describing dataset

3. Training classification models

4. Explain classification models
