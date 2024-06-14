import h5py

def print_hdf5_dimensions(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Function to recursively print dataset dimensions
        def print_dataset_dimensions(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Dimensions: {obj.shape}")

        # Traverse the file and print dimensions of each dataset
        file.visititems(print_dataset_dimensions)

# Path to your HDF5 file
file_path = 'features/conv2/features.hdf5'
print_hdf5_dimensions(file_path)
