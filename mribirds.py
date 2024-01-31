# import packages
import os

from tqdm import tqdm
import numpy as np
import sklearn.model_selection as skms
import sklearn.metrics as skmt

import torch
import torch.utils.data as td
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF



# define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
OUT_DIR = 'results'
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

in_dir_data = 'mribirdsdata'

# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)


def get_model_desc(pretrained=False, num_classes=200, use_attention=False):
    """
    Generates description string.  
    """
    desc = list()

    if pretrained:
        desc.append('Transfer')
    else:
        desc.append('Baseline')

    if num_classes == 204:
        desc.append('Multitask')

    if use_attention:
        desc.append('Attention')

    return '-'.join(desc)


def save_accuracy(path_to_csv, desc, acc, sep='\t', newline='\n'):
    """
    Logs accuracy into a CSV-file.
    """
    file_exists = os.path.exists(path_to_csv)

    mode = 'a'
    if not file_exists:
        mode += '+'

    with open(path_to_csv, mode) as csv:
        if not file_exists:
            csv.write(f'setup{sep}accuracy{newline}')

        csv.write(f'{desc}{sep}{acc}{newline}')


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
params = {'batch_size': 24, 'num_workers': 2}
num_epochs = 10
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

# instantiate optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# generate model description string
model_desc = get_model_desc(num_classes=num_classes, pretrained=pretrained)

# define the training loop
best_snapshot_path = None
val_acc_avg = list()
best_val_acc = -1.0

for epoch in range(num_epochs):
    
    # train the model
    model.train()
    train_loss = list()
    i = 0
    for batch in tqdm(train_loader):
        i += 1
        x, y = batch
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # calculate the loss
        y_pred = model(x)
        
        # calculate the loss
        loss = F.cross_entropy(y_pred, y)
        
        # backprop & update weights 
        loss.backward()
        
        optimizer.step()

        train_loss.append(loss.item())
        
    # validate the model
    model.eval()
    val_loss = list()
    val_acc = list()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # predict bird species
            y_pred = model(x)

            # calculate the loss
            loss = F.cross_entropy(y_pred, y)
            
            # calculate the accuracy
            acc = skmt.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

            val_loss.append(loss.item())
            val_acc.append(acc)

        val_acc_avg.append(np.mean(val_acc))
            
        # save the best model snapshot
        current_val_acc = val_acc_avg[-1]
        if current_val_acc > best_val_acc:
            if best_snapshot_path is not None:
                os.remove(best_snapshot_path)

            best_val_acc = current_val_acc
            best_snapshot_path = os.path.join(OUT_DIR, f'model_{model_desc}_ep={epoch}_acc={best_val_acc}.pt')

            torch.save(model.state_dict(), best_snapshot_path)

    # adjust the learning rate
    scheduler.step()

    # print performance metrics
    print('Epoch {} |> Train. loss: {:.4f} | Val. loss: {:.4f}'.format(
        epoch + 1, np.mean(train_loss), np.mean(val_loss))
    )
        
# use the best model snapshot
model.load_state_dict(torch.load(best_snapshot_path, map_location=DEVICE))
        
# test the model
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

# calculate the accuracy 
test_accuracy = skmt.accuracy_score(true, pred)

# save the accuracy
path_to_logs = f'{OUT_DIR}/logs.csv'
save_accuracy(path_to_logs, model_desc, test_accuracy)

print('Test accuracy: {:.3f}'.format(test_accuracy))
