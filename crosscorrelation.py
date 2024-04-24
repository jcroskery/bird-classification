import h5py
import cupy as cp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def spearman_correlation(x, y):
    n = len(x)
    rank_x = cp.argsort(x)
    rank_y = cp.argsort(y)
    rank_mean_x = cp.argsort(rank_x)
    rank_mean_y = cp.argsort(rank_y)
    
    diff_rank_x = rank_mean_x - cp.arange(n)
    diff_rank_y = rank_mean_y - cp.arange(n)
    
    d_squared = cp.sum(diff_rank_x ** 2)
    spearman_corr = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    
    return spearman_corr

def compute_correlation(i, j, data_gpu):
    return spearman_correlation(data_gpu[i], data_gpu[j])

def main(hdf5_file_path, dataset_name, save_file=None):
    # Open HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        data = f[dataset_name][:]
    
    # Transfer data to GPU
    data_gpu = cp.asarray(data)
    
    num_rows = data.shape[0]
    correlations = cp.zeros((num_rows, num_rows))
    
    # Calculate upper triangle of correlations matrix in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        with tqdm(total=num_rows * (num_rows + 1) // 2, desc="Calculating correlations", unit="pairs") as pbar:
            for i in range(num_rows):
                for j in range(i+1):
                    futures.append(executor.submit(compute_correlation, i, j, data_gpu))
                    pbar.update(1)
        
        # Retrieve results and fill upper triangle of correlations matrix
        for i in range(num_rows):
            for j in range(i+1):
                correlations[i, j] = futures.pop(0).result()
    
    # Fill lower triangle of correlations matrix
    correlations = cp.triu(correlations) + cp.triu(correlations, 1).T
    
    # Transfer correlations back to CPU
    correlations_cpu = cp.asnumpy(correlations)
    
    # Save the correlation matrix as an image file using matplotlib
    if save_file:
        plt.imshow(correlations_cpu, cmap='viridis')
        plt.colorbar()
        plt.savefig(save_file)
    
    return correlations_cpu

if __name__ == "__main__":
    hdf5_file_path = "./features/fc/features.hdf5"
    dataset_name = 'features'
    save_file = "correlation_matrix_fc.png"
    correlations = main(hdf5_file_path, dataset_name, save_file)
    print("Correlation matrix saved as:", save_file)
