# Reporting and Visualization

## Reporting
Once the training is complete, results are stored in the outputs/<exp_name> directory, where <exp_name> is determined based on the experiment's hyperparameters.

Results are organized into the following folders:
- **data**: Contains the data points used for training, validation, and testing, organized by clients and tasks. Each client has separate CSV files for train, validation, test, and holdout datasets. Similarly, each task within a client has its own CSV files for these datasets.
- **logs**: Contains metric results for training, validation, and testing in CSV format.
- **plots**: Contains plots of the validation and test results. Note that these results report a matrix of performance over tasks. In the paper, we report the cumulative performance over tasks, which is obtained by aggregating scores from this matrix. This means the matrix itself is not the reported score but serves as the basis for the aggregated results.
- **visualization**: Contains visualizations of dataset splits by clients and tasks, categorized by target or sensitive attributes.

## Plot Generation (Optional)
We provide scripts for generating plots similar to those in the paper, located in the `results_processing` directory. These scripts create visualizations by combining results from specified algorithms using data from the `outputs` directory.

### Common Instructions for All Plot Scripts
To generate plots using any of the provided scripts:
1. **Navigate to the `results_processing` directory:**
    ```bash
    cd results_processing
    ```
2. **View available customization options for the script:**
    ```bash
    python [script_name].py --help
    ```


### Main Experiments (Test Metrics)
For plots related to main experiments, use the `generate_test_metrics_plot.py` script.
- **Example command** to generate plots for two datasets and two algorithms:
    ```bash
    python generate_test_metrics_plot.py --algorithms ERM fedavg --datasets COVID CheXpert --exp_base_path outputs --metric_name test_per_category_acc --num_tasks 4 --num_clients 10 --num_chexpert_clients 50 --num_rounds 150 --num_iters 5
    ```

### Main Experiments (Holdout Metrics)
For visualizing metrics on holdout data, use the `generate_holdout_metrics_bar.py` script.
- **Example command** to generate holdout metric bars for specific algorithms across multiple datasets:
    ```bash
    python generate_holdout_metrics_bar.py --algorithms ERM fedavg --datasets COVID CheXpert --exp_base_path outputs --metric_name holdout_per_category_acc --num_tasks 4 --num_clients 10 --num_chexpert_clients 50 --num_rounds 150 --num_iters 5
    ```

### Novel Disease Experiments
For specialized plots examining novel disease scenarios, use `generate_novel_plot.py`.
- **Example command**:
    ```bash
    python generate_novel_plot.py --algorithms ERM fedavg --exp_base_path outputs --metric_name holdout_per_category_acc --num_tasks 4 --num_clients 5 --num_rounds 150 --num_iters 5
    ```



## Reading and Analyzing Experiment Results

For users who need to directly analyze the results of experiments outside the predefined plotting scripts, we offer the `read_exp_results.py` utility. This script allows for straightforward computation of statistical metrics like mean and standard deviation across client accuracies from specified experiment paths.

The `read_exp_results.py` script is designed to quickly aggregate and report statistics from multiple experiment outputs. Here’s how to use it:

1. **Modify the Script to Include Desired Paths:**
   - Before running the script, specify the paths to the experiment results within the `experiment_paths` list. These paths should point to directories where the results of various experiments are stored.

2. **Run the Script:**
   - Navigate to the directory containing the script and run it:
     ```bash
     cd results_processing
     python read_exp_results.py
     ```
