# coding:utf-8
import numpy as np
import random 
from PIL import Image 
import torch 
from torch.autograd import Variable
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 

def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): 
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img)) 
        img.save('./save_path' + weight_name + '_' + image_name + '.png')   ###  your save path  ###

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc
    
def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
    
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []

        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
                    
        return Variable(torch.cat(to_return))  

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
        
        
