from django.urls import path,re_path
from datasets.views import DatasetsMainView, create_dataset
from django.contrib.auth.decorators import login_required
import datasets.views as dataset_views

from datasets.views import category_list

urlpatterns = [
    #path('', login_required(DatasetsMainView.as_view(),login_url='/login/')),
    path('', DatasetsMainView.as_view()),
    path('<int:ex_id>/', dataset_views.detail_dataset, name='detail'),

    path('<int:dataset>/<category>/', dataset_views.category_view, name='detail'),
    path('<int:dt_id>/<str:category>/<int:img_id>/', dataset_views.detail_image, name='detail_image'),

    #path('create_dataset/', create_dataset),
    #('<int:ex_id>/edit', dataset_views.edit_dataset, name='edit'),

]

urlpatterns += []