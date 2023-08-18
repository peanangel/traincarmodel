import cv2
import requests
import numpy as np
import pickle
import os
import base64

url = "http://localhost:8080/api/genhog"
def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})
    return response.json()

path = '..\\train\\'
carvectors = []
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path,sub)):
        img_file_name = os.path.join(path,sub)+"/"+fn
        img = cv2.imread(img_file_name)
        res = img2vec(img)
        vec = list(res["message"])
        vec.append(sub)
        carvectors.append(vec)
write_path = "carvectors_train.pkl"
pickle.dump(carvectors, open(write_path,"wb"))
print("data preparation is done")