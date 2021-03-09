import os
import requests

class RecomAPI():

    def __init__(self,endpoint_name,port = 5000):
        self.url = f"http://rnd-3.eanadev.org:{port}/{endpoint_name}"

    def is_active(self):
        try:
            requests.get(self.url).json() 
            active = True
        except:
            active = False

        return active

    def encoding_dict(self):
        encoding_dict = None
        if self.is_active():
            resp = requests.get(self.url).json()
            encoding_dict = resp['encoding_dict']
        return encoding_dict
        
    def predict(self,img_path_list):
        if not self.is_active():
            return None

        if isinstance(img_path_list,str):
            img_path_list = [img_path_list]

        pred_dict = {}
        bs = 10

        path = img_path_list[0]

        file_dict = {f'file':open(path,'rb')}
        resp = requests.post(self.url,files=file_dict).json() 
        #print(resp)
        pred_dict.update(resp)

        return pred_dict




