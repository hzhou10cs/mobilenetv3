import numpy
import torch
from torchsummary import summary
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from ptflops import get_model_complexity_info
import json
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from jtop import jtop

target_model = 'small'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

if target_model == 'small':
    # MobileNetV3_Small
    model = MobileNetV3_Small().to(device)
    model.load_state_dict(torch.load("450_act3_mobilenetv3_small.pth", map_location=device))
else:
    # MobileNetV3_Large
    model = MobileNetV3_Large().to(device)
    model.load_state_dict(torch.load("450_act3_mobilenetv3_large.pth", map_location=device))
    print('Loading base network...')

model.eval()
print(type(model))
# print(summary(model, (3,224,224),device='cuda'))

#macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True)
#print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# dataset = foz.load_zoo_dataset("imagenet-sample")
# session = fo.launch_app(dataset)
# print(type(dataset))

# Load the labels.json file
labels_file_path = './imagenet-sample/labels.json'
with open(labels_file_path, 'r') as file:
   label_dict = json.load(file)
   classes = label_dict['classes']
   labels = label_dict['labels']

class ImageNetSubset(Dataset):
    def __init__(self, image_folder, labels, transform=None):
        """
        Custom dataset for loading ImageNet subset images.
        :param image_folder: Path to the folder where images are stored.
        :param labels: Dictionary mapping image file names to class labels.
        :param transform: Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_file = list(self.labels.keys())[idx] + '.jpg'
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        label = self.labels[image_file[:-4]]  # Remove '.jpg' to match key

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Initialize the dataset
image_folder = './imagenet-sample/data'  # Update this path if your images are stored elsewhere
dataset = ImageNetSubset(image_folder=image_folder, labels=labels, transform=transform)

# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Adjust batch_size as needed

# Example usage
accuracy = 0
start_time = time.time()

print("Begining inference")
# with torch.no_grad():
#     for i, (images, gt_labels) in enumerate(dataloader):
#         images, gt_labels = images.to(device), gt_labels.to(device)
#         # warming up
#         # if i == 0:
#         #     for _ in range(1000):
#         #         output, inference_times = model(images)
#         #     print("finish warm up ....")

#         output, inference_times = model(images)
#         _, pred_labels = torch.max(output,1)
#         if gt_labels == pred_labels:
#             accuracy += 1

#         print(f"Inference times for image {i+1}:")
#         total_time = 0
#         for part, time_taken in inference_times.items():
#             print(f"{part}: {time_taken:.6f} seconds")
#             total_time += time_taken
#         print(f"Total inference times is {total_time:.6f} seconds")

#         break  # Break here just to demonstrate; remove this in your actual testing loop

# total_inference_time = time.time()-start_time
# print(f"Total time duration of 1000 inferences: {total_inference_time:.6f} seconds")
# accuracy = accuracy/len(dataloader)
# print("Accuracy for 1000 classes: ", accuracy)

iter_times = 10
data_record = {}
mode_record = 'latency'
with torch.no_grad():
    data_num = 0
    #------------- WARM UP ----------------
    for i, (images, gt_labels) in enumerate(dataloader):
            images, gt_labels = images.to(device), gt_labels.to(device)
            _, _ = model(images, mode_record)
    #------------- RECORD START -----------
    for iter_time in range(iter_times):
        for i, (images, gt_labels) in enumerate(dataloader):
            images, gt_labels = images.to(device), gt_labels.to(device)
            # warming up
            # if i == 0:
            #     for _ in range(1000):
            #         output, inference_times = model(images)
            #     print("finish warm up ....")

            output, inference_times = model(images, mode_record)
            _, pred_labels = torch.max(output,1)
            if gt_labels == pred_labels:
                accuracy += 1

            # print(f"Inference times for image {i+1}:")
            total_time = sum(inference_times.values())
            print(f"Total inference times is {total_time:.6f} seconds")
            inference_times["total_time"] = total_time
            data_record[data_num] = inference_times
            data_num += 1

accuracy = accuracy/len(dataloader)
print("Accuracy for 1000 classes: ", accuracy)

with open('tx2_inf_latency.json','w') as fp:
    json.dump(data_record, fp)