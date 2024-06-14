import h5py
import numpy as np
import matplotlib.pyplot as plt

layer = 'fc'

# Function to read a dataset from an HDF5 file
def read_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, 'r') as file:
        dataset = file[dataset_name][:]
    return dataset

# Function to perform correlation using numpy.corrcoef
def correlate_matrix(matrix):
    # Compute the correlation coefficient matrix
    corr_matrix = np.corrcoef(matrix)
    np.fill_diagonal(corr_matrix, np.nan)
    return corr_matrix

# Path to your HDF5 file and dataset name
file_path = 'features/' + layer + '/features.hdf5'
dataset_name = 'features'  # The dataset is called features

# Read the matrix from the HDF5 file
matrix = read_hdf5_dataset(file_path, dataset_name)

# Perform correlation
result = correlate_matrix(matrix)

# Print the result
print("Correlation coefficient matrix:\n", result)

# Save a plot
plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Coefficient Matrix')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.savefig('features/' + layer + '/' + layer + 'matrix.png')  # Save the plot to a file
plt.close()

# Save the matrix to a file
np.savetxt('features/' + layer + '/' + layer + 'matrix.csv', result, delimiter=',', fmt='%.5f')
