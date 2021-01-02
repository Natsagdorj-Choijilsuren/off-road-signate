import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


categories = {
    'road': [128, 64, 128],
    'dirt road': [255, 128, 128],
    'parking lot': [190, 153, 153],
    'driveway': [102, 102, 156],
    'other road marking': [152, 251, 152],
    'rail': [255, 0, 0],
    'lane boundary': [0, 60, 100],
    'other lane marking': [0, 80, 100],
    'puddle': [120, 240, 120],
    'rut': [128, 0, 255],
    'other obstacles': [0, 0, 70],
    'sky': [70, 130, 180],
    'person': [220, 20, 60],
    'two wheel venicle': [119, 11, 32],
    'car': [0, 0, 142],
    'traffic sign': [220, 220, 0],
    'building': [70, 70, 70],
    'crack': [90, 120, 130],
    'snow': [255, 255, 255]
}


classes = {
    'background': 0,
    'road': 1, 
    'dirt road': 2,
    'parking lot': 3,
    'driveway': 4,
    'other road marking': 5,
    'rail': 6, 
    'lane boundary': 7,
    'other lane marking': 8, 
    'puddle': 9, 
    'rut': 10,
    'other obstacles': 11, 
    'sky': 12,
    'person': 13, 
    'two wheel venicle': 14,
    'car': 15, 
    'traffic sign': 16,
    'building': 17,
    'crack': 18,
    'snow': 19
}


def get_number_file(image_name):
    return image_name.split('_')[-1]


class TrainLoader(Dataset):
    
    def __init__(self, image_folder, anno_folder, transform):
        
        self.image_root = image_folder
        self.anno_root = anno_folder
        self.A_folders = ['train_images_A_0', 'train_images_A_1', 'train_images_A_2']
        self.B_folders = ['train_images_B_0', 'train_images_B_1', 'train_images_B_2']

        self.image_width = int(1080/5)
        self.image_height = int(1980/5)
        
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

        '''
        for b_fold in self.B_folders:
            image_folder = os.path.join(self.image_root, b_fold)
            
            for image_name in os.listdir(image_folder):
                if not image_name.endswith('.png'):
                    continue
                
                number_str = get_number_file(image_name)
                
                anno_path = os.path.join(anno_folder_B, 'train_annotation_' + number_str)
                image_path = os.path.join(image_folder, image_name)
                image_paths.append((image_path, anno_path))
        '''
        return  image_paths

    def __len__(self):
        return len(self.paths)
                
    def __getitem__(self, idx):
        img_path, label_path = self.paths[idx]
        
        img, label = TrainLoader.preprocess_image(img_path,
                                                  label_path)        
        if self.transform:
            img = self.transform(img)

        return (img, label)

    @staticmethod
    def preprocess_image(img_path, label_path):
        
        img = Image.open(img_path)
        label = Image.open(label_path)

        img = img.resize((self.image_width, self.image_height),
                         Image.ANTIALIAS)
        label = label.resize((self.image_width, self.image_height),
                             Image.NEAREST)
        
        
        ret_array = np.zeros((self.image_width, self.image_height), dtype=np.int64)
        for category in categories:
            #x, y = np.where(np.array(pil_image))
            x, y = np.where((np.array(pil_image)==categories[category]).sum(axis=2)==3)
            ret_array[x, y] = classes[category]

        return img, ret_array

    
def get_loader(image_folder, anno_folder, num_batches):
    dataset_torch = TrainLoader(image_folder, anno_folder, transform=transform)
    loader = DataLoader(dataset_torch, batch_size=num_batches, shuffle=True,
                        num_workers=4)
                        
    return loader


if __name__ == '__main__':

    loader = get_loader(image_folder='../image_data', anno_folder='../annotation_data',
                        num_batches=20)

    
    #batch_x, batch_y = next(iter(loader))

    print (len(loader))

