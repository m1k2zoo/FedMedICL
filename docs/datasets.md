# Datasets

## Downloading Datasets
Due to data use agreements, direct download links for datasets cannot be provided. Please use the links provided in the table below to register and access the datasets. This table is adapted from the [MEDFAIR GitHub Repository](https://github.com/ys-zong/MEDFAIR).

| Dataset       | Access                                                                                                              |
|---------------|---------------------------------------------------------------------------------------------------------------------|
| **CheXpert**  | [Original data](https://stanfordmlgroup.github.io/competitions/chexpert/) <br> [Demographic data](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) |
| **COVID**  | [Access here](https://github.com/ieee8023/covid-chestxray-dataset)                                                        |
| **Fitzpatrick17k** | [Access here](https://github.com/mattgroh/fitzpatrick17k)                                                         |
| **HAM10000**  | [Access here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)                      |
| **PAPILA**    | [Access here](https://www.nature.com/articles/s41597-022-01388-1#Sec6)                                              |
| **OL3I** | [Access here](https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc)

After downloading the datasets, update the `path_to_dataset` property for each dataset in the `configs/datasets.json` file to point to the corresponding metadata CSV file of the dataset.

For Fitzpatrick17k, we provide a script that downloads the required images based on the metadata from fitzpatrick17k.csv found in the link above. To run this script, execute:
```bash
cd preprocessing
python download_fitzpatrick17.py
```

## Preprocessing Datasets
 Once the paths are set, execute the following command to preprocess the data:
```bash
cd preprocessing
python run_all.py
```
This script processes all specified datasets and produces training, validation, and testing splits. It also updates the properties "train_meta_path", "val_meta_path", and "test_meta_path" in confgis/datasets.json.

