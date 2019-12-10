#Initialize pointers
data_set = 'VOC'
dataset_root = '//datasets/home/12/312/vpotnuru/data/VOC0712/VOC2007/'
voc_root = '//datasets/home/12/312/vpotnuru/data/VOC0712/'
save_folder = 'trained_weights/'
eval_save_folder = 'eval/'
devkit_path = 'devkit_path/'
output_dir = "out/"

#Run related metaparameters

batch_size = 32
resume = None

#Optimization metaparameters
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
    
confidence_threshold = 0.01
top_k = 5
cleanup = True

YEAR = '2012'
dataset_mean = (104, 117, 123)
set_type = 'train'

# models to start from , based on he problem 
trained_model = 'weights/ssd_pretrained.pth'
basenet = 'weights/vgg16_reducedfc.pth'
