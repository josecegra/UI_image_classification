from django.db import models



# Create your models here.
class DatasetModel(models.Model):

    name = models.CharField(max_length=255, default = '')

    categories = models.CharField(max_length=5000, default = '',blank=True)
    #username = models.CharField(max_length=255, default = '')
    #id = models.AutoField(primary_key=True,default=1)

    # CHOICES = [
    # ('none','none'),
    # ('classification','classification'),
    # ('segmentation','segmentation'),
    # ]

    # problem_type = models.CharField(
    #     max_length=20,
    #     choices=CHOICES,
    #     default='none',
    # )

    images_path = models.CharField(max_length=255, default = '')
    #annotations_path = models.CharField(max_length=255, default = '')
    #annotations_upload = models.FileField()
    #is_public = models.BooleanField(default=False)
    #img_list = models.ManyToManyField(ImageModel)

    def __str__(self):
        return str(self.name)

class ImageModel(models.Model):
    id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=255, default = '')
    #img_file = models.FileField()
    category = models.CharField(max_length=255, default = '')
    img_url = models.CharField(max_length=500, default = '')
    #dataset = models.ForeignKey(DatasetModel,on_delete=models.CASCADE)
    dataset = models.ManyToManyField(DatasetModel)

    def __str__(self):
        return str(self.filename)




