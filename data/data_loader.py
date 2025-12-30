import os
import pandas as pd
from typing import Optional, Union


def load_ham10000_data(base_path: Union[str, os.PathLike]) -> pd.DataFrame:
    print('Loading HAM10000 dataset....')
    base_path = str(base_path)
    
    metadata_path = os.path.join(base_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    def get_image_path(image_id: str) -> Optional[str]:
        part1 = os.path.join(base_path, 'HAM10000_images_part_1', f'{image_id}.jpg')
        part2 = os.path.join(base_path, 'HAM10000_images_part_2', f'{image_id}.jpg')
        if os.path.exists(part1):
            return part1
        if os.path.exists(part2):
            return part2
        return None
    
    df['image_path'] = df['image_id'].apply(get_image_path)
    df = df[df['image_path'].notna()].reset_index(drop=True)
    df['binary_label'] = (df['dx'] == 'mel').astype(int)
    
    print(f"Loaded {len(df)} images")
    print(f"Melanoma: {df['binary_label'].sum()}")
    print(f"Benign: {len(df) - df['binary_label'].sum()}")
    print(f"\nClass Distribution:")
    print(df['dx'].value_counts())
    
    return df