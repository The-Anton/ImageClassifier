import argparse
import json
import torch
import PIL
from PIL import Image
import numpy as np
from math import ceil
from torchvision import models


parser = argparse.ArgumentParser()

parser.add_argument('--image', type=str,
                    help='path of image file for prediction', required=True)

parser.add_argument('--checkpoint',type=str,
                    help='checkpoint file path for loading model states',required=True)
    
parser.add_argument('--top_k',type=int, default=3,
                    help='number of top predictions')
    
parser.add_argument('--category_names',type=str, default= "cat_to_name.json",
                    help='mapping of categories to real names')

parser.add_argument('--gpu',action="store_true",
                    help='use gpu for commputation')

in_arg = parser.parse_args()
    

    
    

with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
checkpointData_ = torch.load(in_arg.checkpoint, map_location=('cuda' if ( torch.cuda.is_available()) else 'cpu'))
        
    
if checkpointData_['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
else: 
    exec("model = models.{}(pretrained=True)".checkpointData_['arch'])
    model.name = checkpointData_['arch']
    
for param in model.parameters(): 
    param.requires_grad = False
    
model.classifier = checkpointData_['classifier']
model.class_to_idx = checkpointData_['class_to_idx']
model.load_state_dict(checkpointData_['state_dict'])
    

def process_image(image_path):
    short_size= 256
    img_new = Image.open(image_path)
    
    if img_new.size[0] > img_new.size[1]:
        img_new.thumbnail((20000, short_size))
    else:
        img_new.thumbnail((short_size, 20000))
            
    l_margin = (img_new.width-224)/2
    b_margin = (img_new.height-224)/2
    r_margin = l_margin + 224
    t_margin = b_margin + 224
    img_new = img_new.crop((l_margin, b_margin, r_margin,t_margin))
    img_new=np.array(img_new)/255
    img_new= (img_new-np.array([0.485, 0.456, 0.406]))/(np.array([0.229, 0.224, 0.225]))
    img_new = img_new.transpose((2,0,1))

    return img_new



def gpu_status():
    if not in_arg.gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("Failed to find cuda going with cpu")
    return device




def predict(image_path, model, topk):
    
    device = gpu_status()        
    img = process_image(image_path)            
    inputs = torch.from_numpy(np.expand_dims(img, axis=0)).type(torch.FloatTensor)
    model.cpu()
    output = model.forward(inputs)
    pb= torch.exp(output)
    
    top_pb, top_class = pb.topk(topk)
    top_pb = top_pb.tolist()[0]
    top_class = top_class.tolist()[0]
    
    

    data = {val: key for key, val in model.class_to_idx.items()}
   
    top_flow = []
    
    for i in top_class:
        i_ = "{}".format(data.get(i))
        top_flow.append(cat_to_name.get(i_))
        
    return top_pb, top_flow
    
    
data = predict(in_arg.image, model, in_arg.top_k)

probs, flowers = data
for iterate in range(in_arg.top_k):
   
    if iterate+1 ==1:
        print("{} is the most likely flower with {}% liklihood".format(flowers[iterate],ceil(probs[iterate]*100)))

    else:
        print("{} flower with {}% liklihood".format(flowers[iterate],ceil(probs[iterate]*100)))
