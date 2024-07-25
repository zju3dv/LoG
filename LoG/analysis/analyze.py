import matplotlib.pyplot as plt
import seaborn as sns   
import os
import numpy as np

#use histogram to ananlysis and save fig ,input must be 1darray, value_type  as string
def histogram_density_analysis(d_array, value_type, batch_idx):
    # print("------ analysis "+value_type+" ------")
    # Basic statistics
    mean_val = np.mean(d_array)
    median_val = np.median(d_array)
    std_dev = np.std(d_array)
    min_val = np.min(d_array)
    max_val = np.max(d_array)

    # print(f"Mean: {mean_val}")
    # print(f"Median: {median_val}")
    # print(f"Standard Deviation: {std_dev}")
    # print(f"Min: {min_val}")
    # print(f"Max: {max_val}")

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

    analysis_outdir="analysis_image/image_info"
    plt.savefig(os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
    # print("save in "+os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
    # print("------ analysis "+value_type+" ------")


def analyze_hist(output,batch_idx):
    # range_size analysis
    np_range_size=output['range_size'][0].cpu().numpy().reshape(-1,)
    histogram_density_analysis(np_range_size,'range_size',batch_idx)

    # get the max depth
    primitive_idx=output['ranges'][0].cpu().numpy()[:,:,1].reshape(-1,)-1
    fatherest_depth_index=output['primitive_index'][0].cpu().numpy()[primitive_idx]
    #use fatherest_depth_index as index to access the output['depth'][0]
    depth = output['point_depth'][0].cpu().numpy()
    depth = depth.reshape(-1)

    selected_depth = depth[fatherest_depth_index]
    histogram_density_analysis(selected_depth,'max_depth',batch_idx)

    #analysis the depth
    np_range_size=output['point_depth'][0].cpu().numpy().reshape(-1,)
    histogram_density_analysis(np_range_size,'depth',batch_idx)

def summary_analyze(output,batch_idx):
    # range_size analysis
    np_range_size=output['range_size']
    histogram_line_analysis(np_range_size,'range_size')

    # get the max depth
    np_range_size=output['max_depth']
    histogram_line_analysis(np_range_size,'max_depth')

    #analysis the depth
    np_range_size=output['point_depth']
    histogram_line_analysis(np_range_size,'depth')

def histogram_line_analysis(d_array, value_type, batch_idx=None):
    # print("------ analysis "+value_type+" ------")

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Histogram
    axs[0].bar(np.arange(len(d_array)),d_array, color='blue', alpha=0.7, edgecolor='black')
    axs[0].set_title(value_type + ' Value Distribution')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    # Density Plot (Optional)
    axs[1].plot(np.arange(len(d_array)),d_array)
    axs[1].set_title(value_type + ' Density Plot of Values')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()

    analysis_outdir="analysis_image/sum_info"

    if batch_idx is None:
        plt.savefig(os.path.join(analysis_outdir, value_type+'_distribution.png'))
        # print("save in "+os.path.join(analysis_outdir, value_type+'_distribution.png'))
    else:
        plt.savefig(os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
    #     print("save in "+os.path.join(analysis_outdir, value_type+'_distribution_%04d.png'%(batch_idx)))
    # print("------ analysis "+value_type+" ------")