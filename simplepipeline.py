import cv2 
from models.scene_extractor import scene_extractor
from models.classifier import resnet50_model
from dataloader.region_loader import region_generator
from executer.inference import Inference
from utils.namesDict import idx_classes 
from executer.visualizeImg import classes_on_img


path = "/home/ahmed/Datasets/Hieroglyph/GlyphDataset/Dataset/Pictures/egyptianTexts3.jpg"
model_path= "/home/ahmed/Datasets/Hieroglyph/classifier/weights/model.pth"
classes_path="configs/classes.json"

names= idx_classes(classes_path)

img= cv2.imread(path)
regions= scene_extractor(path)
ims_generator= region_generator(img,regions,64)
model = resnet50_model(model_path)
predictions,regions= Inference(model,ims_generator)

predictions= list(map(lambda pred: names[str(pred)],predictions)) # note if used twice will raise error
classes_on_img(img,predictions,regions)