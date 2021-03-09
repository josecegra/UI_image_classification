# UI Image Classification

`pip install -r requirements.txt`

`cd UI_image_classification`

To do :
Copy data to static folder

Copy your dataset to the static folder of django
`cp -r path/to/your/dataset interface/static/`

Set up interface
`python interface/manage.py runserver 0:8001`


http://{domainName}:8001/datasets


Set up classification model

`python classifier/flask_api.py --model_path classifier/checkpoints/checkpoint.pth --class_index_path classifier/checkpoints/class_index.json --saving_dir interface/static/XAI`


Set up recommendation model

`python visual_recom/flask_api.py --endpoint_name img_recommendation --model_path visual_recom/checkpoints/model_3000.pth --embeddings_path visual_recom/checkpoints/embeddings_3000.csv --data_path interface/static/training_data_3000`