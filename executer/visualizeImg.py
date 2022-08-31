import cv2 

def classes_on_img(img,predictions,regions):
    newimg= img.copy()
    for pred,reg in zip(predictions,regions):
        cv2.rectangle(newimg, (reg[2], reg[0]), (reg[3], reg[1]), (0, 0, 255), 0)
        cv2.putText(img=newimg, text=str(pred), org=(int(reg[2]) ,int(reg[0])),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0,255, 0),thickness=1)

    cv2.imwrite("results/out.png",newimg)
    return 
