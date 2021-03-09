from django.urls import path,re_path
from datasets.views import DatasetsMainView
from django.contrib.auth.decorators import login_required
import datasets.views as dataset_views

from datasets.views import category_list

urlpatterns = [
    path('', DatasetsMainView.as_view()),
    path('<int:ex_id>/', dataset_views.detail_dataset, name='detail_dataset'),
    path('<int:dataset>/<category>/', dataset_views.category_view, name='detail_category'),
    path('<int:dt_id>/<str:category>/<int:img_id>/', dataset_views.detail_image, name='detail_image'),
]

urlpatterns += []