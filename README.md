# UI Image Classification

`pip install -r requirements.txt`

`cd UI_image_classification`

# Django interface

Copy your dataset to the static folder of django
`cp -r path/to/your/dataset interface/static/`

Add the absolute path to the directories `classifier` and `visual_recom` in the script `interface/datasets/views.py`.

Set up interface
`python interface/manage.py runserver 0:8001`

Register your dataset in the admin `http://{domainName}:8001/admin/`

Create a new dataset, adding a name and the absolute path to the root directory of the dataset

Click on Dataset Models
Click on Add dataset model
Specify the name and the absolute path to the root dir of the dataset

The dataset should be visible in `http://{domainName}:8001/datasets`


# Classification model

`python classifier/flask_api.py --model_path classifier/checkpoints/checkpoint.pth --class_index_path classifier/checkpoints/class_index.json --saving_dir interface/static/XAI --endpoint_name classifier`


# Visual recommendation model

`python visual_recom/flask_api.py --endpoint_name img_recommendation --model_path visual_recom/checkpoints/model_3000.pth --embeddings_path visual_recom/checkpoints/embeddings_3000.csv --data_path interface/static/training_data_3000`