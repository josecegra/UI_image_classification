import os
import requests

class ModelAPI():

    def __init__(self,endpoint_name,port = 5000):
        self.url = f"http://localhost:{port}/{endpoint_name}"

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
        
    def predict(self,img_path_list,XAI = False):
        if not self.is_active():
            return None

        if isinstance(img_path_list,str):
            img_path_list = [img_path_list]

        if XAI:
            headers = {'XAI':'true'}
        else:
            headers = {'XAI':'false'}

        pred_dict = {'class_id':{},'class_name':{},'XAI_path':{},'conf':{}}
        bs = 4
        batches = [img_path_list[x:x+bs] for x in range(0, len(img_path_list), bs)]
        for batch in batches:
            file_dict = {f'file{i}':open(path,'rb') for i,path in enumerate(batch)}
            resp = requests.post(self.url, headers = headers, files=file_dict).json() 
            for k in pred_dict.keys():
                pred_dict[k].update(resp[k])

        return pred_dict

# model = ModelAPI('sortifier',port = 5000)
# root_path = 'test_images'
# img_path_list = [os.path.join(root_path,fname) for fname in os.listdir(root_path)]
# pred_dict = model.predict(img_path_list,XAI=False)
# print(pred_dict)


