import torch
import torchvision as tv
from thingsvision import get_extractor_from_model
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
import torchvision.transforms.functional as TF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "results/model_mribirds_frozen=False_finetuned=False_ep=5_acc=0.9861111111111112.pt"

num_classes = 18

model_name = 'resnet50'
source = 'torchvision'
module_name = 'fc'
output = './features/' + module_name
device = DEVICE

model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT).to(DEVICE)

model.fc=torch.nn.Linear(2048,num_classes).to(DEVICE)

model.load_state_dict(torch.load(PATH))
model.eval()

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
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))
transforms_eval = tv.transforms.Compose([
   max_padding,
   tv.transforms.CenterCrop((375, 375)),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

extractor = get_extractor_from_model(
  model=model, 
  device=device,
  backend='pt'
)

root='./mribirdsdata/images' # (e.g., './images/)
batch_size = 1

dataset = ImageDataset(
  root=root,
  out_path=output,
  backend=extractor.get_backend(),
  transforms=transforms_eval
)

batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size, 
  backend=extractor.get_backend()
)

features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True  # flatten 2D feature maps from convolutional layer
)

save_features(features, out_path=output, file_format='hdf5')
