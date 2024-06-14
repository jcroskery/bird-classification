import torch
import torchvision as tv
from thingsvision import get_extractor_from_model
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
import torchvision.transforms.functional as TF

# Layers desired: conv1, layer2.3.conv1, layer3.5.conv1, layer4.0.conv1, layer4.2.conv1, fc

# This determines the layer for extraction
# IMPORTANT: Update this to extract different layers
module_name = 'fc'

# Model to extract layers from
PATH = "results/model_mribirds_frozen=False_finetuned=False_ep=5_acc=0.9861111111111112.pt"

# Image location for confusion (TODO: Update this to the cue images)
root='./mribirdsdata/cueimages' # (e.g., './images/)


batch_size = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 18

model_name = 'resnet50'
source = 'torchvision'
output = './features/' + module_name
device = DEVICE


# Load our custom torchvision model
model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT).to(DEVICE)

model.fc=torch.nn.Linear(2048,num_classes).to(DEVICE)

model.load_state_dict(torch.load(PATH, map_location='cuda:0')) # Ensures that the model is loaded on the first GPU
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

# This loads a ThingsVision extractor from our TorchVision model
extractor = get_extractor_from_model(
  model=model, 
  device=device,
  backend='pt'
)

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
