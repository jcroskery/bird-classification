PATH = "results/model_mribirds_frozen=False_finetuned=False_ep=5_acc=0.9861111111111112.pt"

# import packages
import os

from tqdm import tqdm
import numpy as np
import sklearn.model_selection as skms
import sklearn.metrics as skmt
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data as td
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF

from sklearn.metrics import ConfusionMatrixDisplay 
from matplotlib import pyplot as plt

# define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
RANDOM_SEED = 42

in_dir_data = 'mribirdsdata'


class DatasetBirds(tv.datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')

        super(DatasetBirds, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                indices_to_use.append(int(line.strip('\n')))
                

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        index = 0
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                fn = line.strip('\n')
                if (indices_to_use[index] == 1 and train) or (indices_to_use[index] == 0 and not train):
                    filenames_to_use.add(fn)
                index += 1
                

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(DatasetBirds, self).__getitem__(index)

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target

def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width). 
    Fills up the padded area with value(s) passed to the `fill` parameter. 
    """
    dimensions = tv.transforms.functional.get_image_size(img)
    pad_height = max(0, size_max - dimensions[1])
    pad_width = max(0, size_max - dimensions[0])
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad images to 500 pixels
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))

# transform images
transforms_train = tv.transforms.Compose([
   max_padding,
   tv.transforms.RandomOrder([
       tv.transforms.RandomCrop((375, 375)),
       tv.transforms.RandomHorizontalFlip(),
       tv.transforms.RandomVerticalFlip()
   ]),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_eval = tv.transforms.Compose([
   max_padding,
   tv.transforms.CenterCrop((375, 375)),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# instantiate dataset objects according to the pre-defined splits
ds_train = DatasetBirds(in_dir_data, transform=transforms_train, train=True)
ds_val = DatasetBirds(in_dir_data, transform=transforms_eval, train=True)
ds_test = DatasetBirds(in_dir_data, transform=transforms_eval, train=False)

splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

# set hyper-parameters
params = {'batch_size': 16, 'num_workers': 0}
num_epochs = 100
num_classes = 18
pretrained = True

# instantiate data loaders
train_loader = td.DataLoader(
   dataset=ds_train,
   sampler=td.SubsetRandomSampler(idx_train),
   **params
)
val_loader = td.DataLoader(
   dataset=ds_val,
   sampler=td.SubsetRandomSampler(idx_val),
   **params
)
test_loader = td.DataLoader(dataset=ds_test, **params)

# instantiate the model
model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT).to(DEVICE)

model.fc=torch.nn.Linear(2048,num_classes).to(DEVICE)

model.load_state_dict(torch.load(PATH))
model.eval()

true = list()
pred = list()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_pred = model(x)

        true.extend([val.item() for val in y])
        pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

# reindex = [7, 10, 16, 17, 8, 13, 4, 14, 1, 5, 11, 18, 2, 9, 3, 6, 15, 12] # Shift these down by 1

# reorderedtrue = [0] * len(true)
# reorderedpred = [0] * len(pred)
# for i in range(len(reorderedtrue)):
#     reorderedtrue[i] = reindex[true[i]] - 1
# for i in range(len(reorderedpred)):
#     reorderedpred[i] = reindex[pred[i]] - 1

# reorderedclasses = [0] * len(ds_test.classes)
# for i in range(len(ds_test.classes)):
#     reorderedclasses[i] = ds_test.classes[reindex[i] - 1]

# print(reorderedtrue)
# print(reorderedpred)
# print(reorderedclasses)

test_accuracy = skmt.accuracy_score(true, pred)
print('Test accuracy: {:.3f}'.format(test_accuracy))

confusion = confusion_matrix(true, pred, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=ds_test.classes) 
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation='vertical')
plt.tight_layout()
plt.savefig('confusion.png', pad_inches=5)
