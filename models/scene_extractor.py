import cv2 

def BasicPreprocessing(imgpath):
    img= cv2.imread(imgpath)
    
    if len(img.shape)>2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bluredImg = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.threshold(bluredImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return img,thresh
    
def scene_extractor(imgpath,processing= BasicPreprocessing):
    
    img,thresh= processing(imgpath)
    
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    imglist= [] 
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 400 and area <3000:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            # choose specific width and height
            if (ROI.shape[0] <= 100)and (ROI.shape[1] <= 90):
                coordinates= (y,y+h,x,x+w)
                imglist.append(coordinates)

    return imglist

