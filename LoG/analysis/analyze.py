import matplotlib.pyplot as plt
import seaborn as sns   
import os
import numpy as np

#use histogram to ananlysis and save fig ,input must be 1darray, value_type  as string
def histogram_analysis(d_array, value_type, batch_idx):
        print("------ analysis "+value_type+" ------")
        # Basic statistics
        mean_val = np.mean(d_array)
        median_val = np.median(d_array)
        std_dev = np.std(d_array)
        min_val = np.min(d_array)
        max_val = np.max(d_array)

        print(f"Mean: {mean_val}")
        print(f"Median: {median_val}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")

        # Create a figure and a set of subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Histogram
        axs[0].hist(d_array, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[0].set_title(value_type + ' Value Distribution')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')

        # Density Plot (Optional)
        sns.kdeplot(d_array, bw_adjust=0.5, ax=axs[1])
        axs[1].set_title(value_type + ' Density Plot of Values')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

        analysis_outdir="analysis_image"
        plt.savefig(os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
        print("save in "+os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
        print("------ analysis "+value_type+" ------")
