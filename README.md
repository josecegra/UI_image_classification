# UI Image Classification

pip install -r requirements.txt

Set up classification model

python classifier/flask_api.py --model_path classifier/checkpoints/checkpoint.pth --class_index_path classifier/checkpoints/class_index.json --endpoint_name classifier --host 0.0.0.0 --port 5050 --saving_dir interface/static/XAI

Set up recommendation model


Set up interface
python interface/manage.py runserver 0:8001

http://{domainName}:8001/datasets