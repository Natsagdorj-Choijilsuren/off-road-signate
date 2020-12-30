import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])


def get_number_file(image_name):
    return image_name.split('_')[-1]


class TrainLoader(Dataset):
    
    def __init__(self, image_folder, anno_folder, transform):
        
        self.image_root = image_folder
        self.anno_root = anno_folder
        self.A_folders = ['train_images_A_0', 'train_images_A_1', 'train_images_A_2']
        self.B_folders = ['train_images_B_0', 'train_images_B_1', 'train_images_B_2']
        
        self.transform = transform
        
        self.paths = self.get_file_paths()
        
    def get_file_paths(self):
            
        anno_folder_A = os.path.join(self.anno_root, 'train_annotations_A')
        anno_folder_B = os.path.join(self.anno_root, 'train_annotations_B')
        
        image_paths = []
        for a_fold in self.A_folders:
            image_folder = os.path.join(self.image_root, a_fold)

            for image_name in os.listdir(image_folder):
                if not image_name.endswith('.png'):
                    continue
                
                number_str = get_number_file(image_name)
                anno_path = os.path.join(anno_folder_A, 'train_annotation_' + number_str)
                image_path = os.path.join(image_folder, image_name)
                
                image_paths.append((image_path, anno_path))

        for b_fold in self.B_folders:
            image_folder = os.path.join(self.image_root, b_fold)
            
            for image_name in os.listdir(image_folder):
                if not image_name.endswith('.png'):
                    continue
                
                number_str = get_number_file(image_name)
                
                anno_path = os.path.join(anno_folder_B, 'train_annotation_' + number_str)
                image_path = os.path.join(image_folder, image_name)
                image_paths.append((image_path, anno_path))
                
        return  image_paths

    def __len__(self):
        return len(self.paths)
                
    def __getitem__(self, idx):
        img_path, label_path = self.paths[idx]
        
        img = Image.open(img_path)
        label = Image.open(label_path)
        
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
            
        return (img, label)

    
def get_loader(image_folder, anno_folder, num_batches):
    dataset_torch = TrainLoader(image_folder, anno_folder, transform=transform)
    loader = DataLoader(dataset_torch, batch_size=num_batches, shuffle=True,
                        num_workers=8)
                        
    return loader


if __name__ == '__main__':

    loader = get_loader(image_folder='../image_data', anno_folder='../annotation_data',
                        num_batches=10)

    for x in loader:
        print (x)
