#!/usr/bin/env python3
"""
MegaHan97K Dataloader - Adapted for local dataset paths
Loads Chinese character images from LMDB databases for training.
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
import cv2
import six
import sys
from PIL import Image
import numpy as np


class MegaHanLMDBDataset(Dataset):
    """
    Dataset class for loading MegaHan97K data from LMDB format.
    
    Args:
        root: Path to LMDB database directory
        img_size: Tuple of (height, width) for resizing images
        transform: Optional torchvision transforms to apply
        codebook_path: Path to character codebook file
    """
    
    def __init__(self, root=None, img_size=(96, 96), transform=None, codebook_path=None):
        if not os.path.exists(root):
            raise ValueError(f"LMDB database not found at: {root}")
        
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=1099511627776  # 1TB
        )
        
        if not self.env:
            print(f'Cannot create LMDB from {root}')
            sys.exit(0)
            
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples
        
        self.root = root
        self.transform = transform
        self.img_size = img_size
        self.toTensor = transforms.ToTensor()
        
        # Load codebook if provided
        self.codebook_dict = None
        if codebook_path and os.path.exists(codebook_path):
            self.codebook_dict = self._load_codebook(codebook_path)
    
    def _load_codebook(self, codebook_path):
        """Load character codebook mapping label_id -> character"""
        codebook_dict = {}
        with open(codebook_path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                char = line.strip().split(':')[0]
                codebook_dict[i] = char
        return codebook_dict
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index < len(self), 'Index range error'
        index += 1  # LMDB uses 1-based indexing
        
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            label_key = 'label-%09d' % index
            
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print(f'Corrupted image for index {index}')
                return self[index + 1] if index < len(self) else self[0]
            
            label = str(txn.get(label_key.encode()).decode())
        
        # Resize image
        img = np.asarray(img)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        if self.transform:
            img = self.transform(img)
        
        label = int(label)
        
        # Return with character if codebook available
        if self.codebook_dict:
            char = self.codebook_dict.get(label, '?')
            return (img, label, char)
        
        return (img, label)


def get_megahan_dataloader(
    data_root,
    codebook_path,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    img_size=(96, 96)
):
    """
    Create a DataLoader for MegaHan97K dataset.
    
    Args:
        data_root: Path to LMDB database
        codebook_path: Path to codebook file
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        img_size: Image size (height, width)
    
    Returns:
        DataLoader instance
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MegaHanLMDBDataset(
        root=data_root,
        img_size=img_size,
        transform=data_transform,
        codebook_path=codebook_path
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='MegaHan97K Dataset Loader')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to LMDB database directory')
    parser.add_argument('--codebook', type=str, required=True,
                       help='Path to codebook file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to display')
    parser.add_argument('--save_dir', type=str, default='sample_images',
                       help='Directory to save sample images')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_root}")
    print(f"Using codebook: {args.codebook}")
    
    dataloader = get_megahan_dataloader(
        data_root=args.data_root,
        codebook_path=args.codebook,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} images")
    print(f"Saving {args.num_samples} sample images to {args.save_dir}/\n")
    
    # Save sample images
    import torchvision
    for i, data in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        if len(data) == 3:
            img, label, char = data
            filename = f'{args.save_dir}/{char}_{label.item()}_{i}.jpg'
            print(f"Sample {i+1}: Character '{char}' (label {label.item()})")
        else:
            img, label = data
            filename = f'{args.save_dir}/label_{label.item()}_{i}.jpg'
            print(f"Sample {i+1}: Label {label.item()}")
        
        torchvision.utils.save_image(img, filename, normalize=True)
    
    print(f"\n✓ Saved {min(args.num_samples, len(dataloader.dataset))} samples")
