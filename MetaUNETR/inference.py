from collections import OrderedDict
import os
import glob
import torch
from monai import data
from tokenmixer import tokenmixers
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    EnsureTyped,
    Invertd,
    SaveImaged,
    CenterSpatialCropd,
    Activationsd, 
)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch,Dataset


#--------------------get data and transform-----------------------
data_root= '/dataset/BTCV/RawData/Training/imagesTr/'

train_images = sorted(glob.glob(os.path.join(data_root, '*.nii.gz')))
data_dicts = [{'image': image_name}  for image_name in train_images]

print(len(data_dicts))

output_dir = 'BTCV_test_CNN_re'

test_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),                
            mode=("bilinear"), 
        ),       
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0, b_max=1, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"), 
    ]
)

post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transform,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir,
                   output_postfix=output_dir, output_ext=".nii.gz", resample=False,separate_folder=False), #在invertd 已经 resample回去了
    ])

test_ds = data.Dataset(data=data_dicts[-2:], transform=test_transform)
test_loader = data.DataLoader(test_ds,batch_size=1)
'''---------------------------- models---------------------------------'''
#------------model1-------------
model = tokenmixers(
encoder = 'large_kernel_Conv',#'Sequencer','large_kernel_Conv', 'swintransformer',fft_mixer,'Vip'
img_size=(96,96,96), #
in_channels=1,
out_channels=14,
depths = (2, 2, 2, 2), # tokenmixer num per layer
num_heads = (3, 6, 12, 24), # for transformer
feature_size = 48,  #base channel
hidden_sizes =[12, 24, 36, 48], #sequencer
segment_dim = [48, 24, 12, 6], #vip
mlp_ratio = 4,
)

model_dict = torch.load("CNN/uxnet.pth")
model.load_state_dict(model_dict)
model.cuda().eval()
print('model weight loaded')
with torch.no_grad():
    
        for i, test_data in enumerate(test_loader):
            images = test_data["image"]
            with torch.cuda.amp.autocast():
                roi_size = (96, 96, 96)
                test_data['pred'] = sliding_window_inference(
                    images, roi_size, 1, model, overlap=0.5,sw_device='cuda',device='cpu')
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
