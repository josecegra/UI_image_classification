from django.shortcuts import render, get_object_or_404, redirect, reverse
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.views.generic import TemplateView
from .models import DatasetModel, ImageModel
from .forms import DatasetForm
from django.core.files.storage import FileSystemStorage
import os
from shutil import copyfile
from django.core.files import File
from visualization.settings import MEDIA_ROOT
import mimetypes

from .api_wrapper import ModelAPI
from .recom_api_wrapper import RecomAPI




category_list = ['building','ceramics']


def get_images(images_path,dataset):
    dest_path = MEDIA_ROOT
    fs = FileSystemStorage()
    img_list = []
    for fname in os.listdir(images_path):
        fpath = os.path.join(images_path,fname)
        dest_file_path = os.path.join(dest_path,fname)
        copyfile(fpath,dest_file_path)
        myfile = fs.open(dest_file_path)  
        uploaded_file_url = fs.url(dest_file_path)   
        obj = ImageModel(filename = fname, img_file = myfile, img_url = uploaded_file_url,dataset = dataset) 
        obj.save()
        img_list.append(obj)
    
    return img_list

class DatasetsMainView(TemplateView):
    template_name = 'datasets/datasets.html'
    def get(self,request):



        #username = request.user.username
        dataset_list = DatasetModel.objects.all()
        #public_dataset_list = DatasetModel.objects.filter(username = 'public')
        context = {'dataset_list':dataset_list,'public_dataset_list':None}

        #context = {'category_list':category_list}



        context.update({'nbar':'datasets','logged':True})
        return render(request, self.template_name,context)

def detail_dataset(request, ex_id):
    dataset = get_object_or_404(DatasetModel, pk=ex_id)
    #img_list = dataset.img_list.all()

    img_list = ImageModel.objects.filter(dataset = ex_id)

    categories = dataset.categories.split(' ')
    
    cat_list = []
    for cat in categories:
        n_images = ImageModel.objects.filter(dataset = ex_id,category = cat).count
        cat_list.append({'category':cat,'n_images':n_images})
    

    

    message = ''
    context = {'cat_list':cat_list,'n_images':n_images,'experiment': dataset,'message':message,'img_list':img_list}
    if request.method == 'POST':
        if 'delete' in request.POST:
            DatasetModel.objects.filter(id=dataset.id).delete()
            return redirect('/datasets/')
        if 'back' in request.POST:
            return redirect('/datasets/')
        if 'edit' in request.POST:
            return redirect(f'/datasets/{ex_id}/edit')

    context.update({'nbar':'datasets','logged':True})
    return render(request, 'datasets/detail_dataset.html', context)



from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def category_view(request,dataset,category):

    if request.method == 'POST':
        if 'back' in request.POST:
            return redirect(f'/datasets/{dataset}')


    dataset = get_object_or_404(DatasetModel, pk=dataset)
    img_obj_list = ImageModel.objects.filter(dataset = dataset.id,category = category)[:]

    img_list = []
    for img_obj in img_obj_list:
        id_enc = img_obj.filename.replace('.jpg','')
        europeana_id = id_enc.replace('[ph]','/')
        img_list.append({'url':img_obj.img_url,'id':europeana_id,'id_enc':id_enc,'img_id':img_obj.id})

    page = request.GET.get('page', 1)

    paginator = Paginator(img_list, 30)
    try:
        img_list = paginator.page(page)
    except PageNotAnInteger:
        img_list = paginator.page(1)
    except EmptyPage:
        img_list = paginator.page(paginator.num_pages)

    # Get the index of the current page
    index = img_list.number - 1  # edited to something easier without index
    # This value is maximum index of your pages, so the last page - 1
    max_index = len(paginator.page_range)
    # You want a range of 7, so lets calculate where to slice the list
    start_index = index - 3 if index >= 3 else 0
    end_index = index + 3 if index <= max_index - 3 else max_index
    # Get our new page range. In the latest versions of Django page_range returns 
    # an iterator. Thus pass it to list, to make our slice possible again.
    page_range = list(paginator.page_range)[start_index:end_index]

    context = {}

    context.update({'page_range':page_range,'img_list':img_list,'category':category,'dataset_id': dataset.id})


    return render(request, 'datasets/category.html',context)


def detail_image(request, dt_id,category,img_id):
    img = get_object_or_404(ImageModel, pk=img_id)
    dataset = get_object_or_404(DatasetModel, pk=dt_id)
    id_enc = img.filename.replace('.jpg','')
    europeana_id = id_enc.replace('[ph]','/')
    context = {'img': img,'dataset':dataset,'category':category,'europeana_id':europeana_id}
    if request.method == 'POST':
        if 'XAI' in request.POST:
            message = 'XAI should happen'
            context.update({'message':message})
            return render(request, 'datasets/detail_image.html', context)
        elif 'refresh' in request.POST:
            context = {'img': img}
            return render(request, 'datasets/detail_image.html', context)
        elif 'board' in request.POST:
            return redirect(f'/datasets/{dt_id}/{category}')
        elif 'delete' in request.POST:
            ImageModel.objects.filter(id=img_id).delete()
            return redirect(f'/datasets/{dt_id}')

        elif 'predict' in request.POST:
            

            #torch_model = experiment.torch_model
            #img_path = img.img_file.path
       
            model_wrapper = ModelAPI('sortifier',5050)

            img_path = os.path.join(dataset.images_path,category,img.filename)
            pred_dict = model_wrapper.predict(img_path,XAI=True)

            if pred_dict:

                for fname in pred_dict['XAI_path'].keys():
                    pass
                #XAI_path = pred_dict['XAI_path'][fname]
                class_name = pred_dict['class_name'][fname]
                conf = round(pred_dict['conf'][fname],3)

                XAI_url = f'/static/XAI/{fname}'

                context.update({'class':class_name,'XAI_url':XAI_url,'conf':conf})

                print(pred_dict)

        elif 'recommend' in request.POST:
            model = RecomAPI('img_recommendation',port = 5000)

            img_path = os.path.join(dataset.images_path,category,img.filename)

            pred_dict = model.predict(img_path)

            #print(pred_dict)
            recom_images = []
            for path in pred_dict['fnames']:
                
                fname = os.path.split(path)[1]
                for img in ImageModel.objects.filter(filename = fname):
                    recom_images.append(img)
                    break

            eu_id_list = []
            for img in recom_images:
                eu_id_list.append(img.filename.replace('[ph]','/').replace('.jpg',''))


            context.update({'recom_imgs':zip(eu_id_list,recom_images)})



        if 'download' in request.POST:
            img = get_object_or_404(ImageModel, pk=img_id)
            img_path = img.img_file.path
            fl = open(img_path, 'rb')
            mime_type, _ = mimetypes.guess_type(img_path)
            response = HttpResponse(fl, content_type=mime_type)
            response['Content-Disposition'] = "attachment; filename=%s" % os.path.split(img_path)[1]
            return response

    context.update({'nbar':'datasets','logged':True})
    return render(request, 'datasets/detail_image.html', context)



def create_dataset(request):
    #username = request.user.username
    if request.method == "POST":
        if 'cancel' in request.POST:
            return redirect('/datasets/')

        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():



            obj = DatasetModel() 
            obj.name = form.cleaned_data['name']
            #obj.problem_type = form.cleaned_data['problem_type']
            obj.images_path = form.cleaned_data['images_path']


            obj.save()

            images_path = form.cleaned_data['images_path']
            img_list = []
            if images_path and os.path.exists(images_path):
                img_list = get_images(images_path,obj)
            #obj.annotations_path = form.cleaned_data['annotations_path']
            #obj.annotations_upload = form.cleaned_data['annotations_upload']

            # if obj.is_public:
            #     obj.username = 'public'
            # else:
            #     obj.username = username
            
            #obj.id = len(DatasetModel.objects.all())+1

            #obj.save()
            
            # for img in img_list:
            #     obj.img_list.add(img.id)

            #obj.save()

            

            
            return HttpResponseRedirect(f'/datasets/{obj.id}')
    else:
        form = DatasetForm()
    context = {'form':form}
    context.update({'nbar':'datasets','logged':True})
    return render(request, 'datasets/create_dataset.html',context)

def edit_dataset(request,ex_id):
    username = request.user.username
    dataset = get_object_or_404(DatasetModel, pk=ex_id)
    if request.method == "POST":
        if 'cancel' in request.POST:
            return redirect(f'/datasets/{ex_id}')

        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():

            images_path = form.cleaned_data['images_path']
            img_list = []
            if images_path and os.path.exists(images_path):
                img_list = get_images(images_path)

            #obj = DatasetModel() 
            dataset.name = form.cleaned_data['name']
            dataset.problem_type = form.cleaned_data['problem_type']
            dataset.images_path = form.cleaned_data['images_path']
            dataset.annotations_path = form.cleaned_data['annotations_path']
            dataset.annotations_upload = form.cleaned_data['annotations_upload']

            if dataset.is_public:
                dataset.username = 'public'
            else:
                dataset.username = username
            #obj.save()

            for img in img_list:
                dataset.img_list.add(img.id)
            form.instance = dataset
            form.save()
            return HttpResponseRedirect(f'/datasets/{dataset.id}')
    else:
        form = DatasetForm(instance=dataset)
        #form.fields['name'] = dataset.name
    context = {'form':form,'dataset':dataset}
    context.update({'nbar':'datasets','logged':True})
    return render(request, 'datasets/edit_dataset.html',context)






# get images when the path is set

    # def post(self,request):
    #     username = request.user.username
    #     fs = FileSystemStorage()
    #     media_dir = os.path.join(settings.BASE_DIR,'media')
    #     path = request.POST.get('path')
    #     if path:
    #         for i,filename in enumerate(os.listdir(path)):
                
    #             src_file_path = os.path.join(path,filename)
    #             dest_file_path = os.path.join(media_dir,filename)
    #             copyfile(src_file_path,dest_file_path)

    #             myfile = fs.open(dest_file_path)  
    #             uploaded_file_url = fs.url(dest_file_path)          
    #             Doc.objects.create(upload = myfile, image_url = uploaded_file_url, user=username)
    #         message = f'Added {i} images'
    #     else:
    #         message = 'Empty path entered'
    #     context = {'message':message}
    #     return render(request, self.template_name,context)