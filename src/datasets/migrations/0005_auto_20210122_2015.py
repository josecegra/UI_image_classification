# Generated by Django 3.1.5 on 2021-01-22 20:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0004_datasetmodel_img_list'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datasetmodel',
            name='img_list',
            field=models.ManyToManyField(blank=True, to='datasets.ImageModel'),
        ),
    ]
