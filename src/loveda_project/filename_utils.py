"""
Utility functions for retrieving image filenames and metadata from LoveDA datasets.

This module provides helper functions to work with filenames when processing
LoveDA images through DataLoaders and training pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import ConcatDataset
from torchgeo.datasets import LoveDA
from loveda_project.data import WrappedLoveDAScene


def _unwrap_dataset(dataset):
    """
    Unwrap a dataset to get the underlying LoveDA or WrappedLoveDAScene.
    
    Handles ConcatDataset, WrappedLoveDAScene, and raw LoveDA instances.
    """
    # If it's a ConcatDataset, we need to handle it differently
    if isinstance(dataset, ConcatDataset):
        # For ConcatDataset, we can access the underlying datasets
        # The first dataset should be a WrappedLoveDAScene
        if hasattr(dataset, 'datasets') and len(dataset.datasets) > 0:
            first_dataset = dataset.datasets[0]
            if isinstance(first_dataset, WrappedLoveDAScene):
                return first_dataset.dataset
            return first_dataset
    
    # If it's a WrappedLoveDAScene, unwrap it
    if isinstance(dataset, WrappedLoveDAScene):
        return dataset.dataset
    
    # Otherwise, assume it's already a LoveDA dataset
    return dataset


def _get_index_in_loveda(dataset, global_index: int) -> Tuple:
    """
    Convert a global index to the LoveDA dataset index.
    
    Returns (loveda_dataset, local_index) tuple.
    """
    if isinstance(dataset, ConcatDataset):
        # Find which sub-dataset this index belongs to
        cumulative = 0
        for sub_dataset in dataset.datasets:
            if global_index < cumulative + len(sub_dataset):
                local_index = global_index - cumulative
                if isinstance(sub_dataset, WrappedLoveDAScene):
                    return sub_dataset.dataset, local_index
                return sub_dataset, local_index
            cumulative += len(sub_dataset)
        raise IndexError(f"Index {global_index} out of range")
    
    # Single dataset
    if isinstance(dataset, WrappedLoveDAScene):
        return dataset.dataset, global_index
    
    return dataset, global_index


def get_image_filename(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], index: int) -> str:
    """
    Get the image filename for a given index in the dataset.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        index: Index of the image
        
    Returns:
        Filename string (e.g., '1366.png')
    """
    loveda_dataset, local_index = _get_index_in_loveda(dataset, index)
    file_dict = loveda_dataset.files[local_index]
    return Path(file_dict['image']).name


def get_image_path(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], index: int) -> str:
    """
    Get the full image path for a given index in the dataset.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        index: Index of the image
        
    Returns:
        Full path to the image file
    """
    loveda_dataset, local_index = _get_index_in_loveda(dataset, index)
    file_dict = loveda_dataset.files[local_index]
    return file_dict['image']


def get_image_stem(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], index: int) -> str:
    """
    Get the image filename without extension (stem).
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        index: Index of the image
        
    Returns:
        Filename stem (e.g., '1366')
    """
    image_path = get_image_path(dataset, index)
    return Path(image_path).stem


def get_mask_path(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], index: int) -> str:
    """
    Get the corresponding mask/label path for a given image index.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        index: Index of the image
        
    Returns:
        Full path to the mask file
    """
    loveda_dataset, local_index = _get_index_in_loveda(dataset, index)
    file_dict = loveda_dataset.files[local_index]
    return file_dict['mask']


def get_file_metadata(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], index: int) -> Dict:
    """
    Get complete file metadata (paths and filenames) for an index.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        index: Index of the image
        
    Returns:
        Dictionary containing:
            - filename: Just the filename (e.g., '1366.png')
            - stem: Filename without extension (e.g., '1366')
            - image_path: Full path to image
            - mask_path: Full path to mask
            - rel_image_path: Relative path from data root
            - rel_mask_path: Relative path from data root
    """
    loveda_dataset, local_index = _get_index_in_loveda(dataset, index)
    
    file_dict = loveda_dataset.files[local_index]
    image_path = file_dict['image']
    mask_path = file_dict['mask']
    
    image_p = Path(image_path)
    mask_p = Path(mask_path)
    
    return {
        'filename': image_p.name,
        'stem': image_p.stem,
        'image_path': image_path,
        'mask_path': mask_path,
        'rel_image_path': str(image_p.relative_to('./data')),
        'rel_mask_path': str(mask_p.relative_to('./data')),
    }


def create_index_to_filename_map(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset]) -> Dict[int, str]:
    """
    Create a complete mapping of dataset indices to filenames.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        
    Returns:
        Dictionary mapping index -> filename
    """
    result = {}
    
    if isinstance(dataset, ConcatDataset):
        # Handle ConcatDataset by iterating through all sub-datasets
        cumulative = 0
        for sub_dataset in dataset.datasets:
            loveda_ds, _ = _get_index_in_loveda(sub_dataset, 0)
            for local_idx, file_dict in enumerate(loveda_ds.files):
                result[cumulative + local_idx] = Path(file_dict['image']).name
            cumulative += len(loveda_ds.files)
    else:
        loveda_dataset = _unwrap_dataset(dataset)
        result = {
            idx: Path(file_dict['image']).name
            for idx, file_dict in enumerate(loveda_dataset.files)
        }
    
    return result


def create_filename_to_index_map(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset]) -> Dict[str, int]:
    """
    Create a reverse mapping of filenames to dataset indices.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        
    Returns:
        Dictionary mapping filename -> index
    """
    idx_to_filename = create_index_to_filename_map(dataset)
    return {filename: idx for idx, filename in idx_to_filename.items()}


def get_all_filenames(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset]) -> List[str]:
    """
    Get a list of all filenames in the dataset.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        
    Returns:
        List of all filenames
    """
    idx_map = create_index_to_filename_map(dataset)
    return [idx_map[i] for i in sorted(idx_map.keys())]


def get_all_image_paths(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset]) -> List[str]:
    """
    Get a list of all image paths in the dataset.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        
    Returns:
        List of all full image paths
    """
    result = []
    
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            loveda_ds, _ = _get_index_in_loveda(sub_dataset, 0)
            result.extend([f['image'] for f in loveda_ds.files])
    else:
        loveda_dataset = _unwrap_dataset(dataset)
        result = [f['image'] for f in loveda_dataset.files]
    
    return result


def get_filenames_in_batch(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], 
                           start_index: int, batch_size: int) -> List[str]:
    """
    Get filenames for all items in a batch.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        start_index: Starting index of the batch
        batch_size: Number of items in the batch
        
    Returns:
        List of filenames in the batch
    """
    filenames = []
    for i in range(batch_size):
        index = start_index + i
        if index < len(dataset):
            filenames.append(get_image_filename(dataset, index))
    return filenames


def filter_by_filename(dataset: Union[LoveDA, WrappedLoveDAScene, ConcatDataset], 
                      filename_pattern: str) -> List[Tuple[int, str]]:
    """
    Find all indices with filenames matching a pattern.
    
    Args:
        dataset: LoveDA, WrappedLoveDAScene, or ConcatDataset
        filename_pattern: Pattern to match (supports wildcards)
        
    Returns:
        List of (index, filename) tuples matching the pattern
    """
    from fnmatch import fnmatch
    matches = []
    
    idx_map = create_index_to_filename_map(dataset)
    for idx in sorted(idx_map.keys()):
        filename = idx_map[idx]
        if fnmatch(filename, filename_pattern):
            matches.append((idx, filename))
    
    return matches


# Example usage functions
def demo_usage():
    """Demonstrate usage of utility functions."""
    from loveda_project.data import LoveDAConfig, build_dataloaders
    
    config = LoveDAConfig(root="./data", patch_size=512)
    scene_datasets, dataloaders = build_dataloaders(config)
    
    # Get the training dataset
    train_dataset = dataloaders["train"].dataset
    
    # Example 1: Get filename for specific index
    print("Example 1: Get filename for index 0")
    filename = get_image_filename(train_dataset, 0)
    print(f"  Filename: {filename}")
    
    # Example 2: Get all metadata
    print("\nExample 2: Get complete file metadata")
    metadata = get_file_metadata(train_dataset, 0)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Example 3: Create index mapping
    print("\nExample 3: Create index-to-filename mapping")
    idx_map = create_index_to_filename_map(train_dataset)
    print(f"  Total files: {len(idx_map)}")
    print(f"  First 3 entries: {dict(list(idx_map.items())[:3])}")
    
    # Example 4: Filter by pattern
    print("\nExample 4: Find all images matching pattern")
    matches = filter_by_filename(train_dataset, "136*.png")
    print(f"  Found {len(matches)} matches")
    for idx, filename in matches[:3]:
        print(f"    [{idx}] {filename}")


if __name__ == "__main__":
    demo_usage()
