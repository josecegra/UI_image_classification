# Generated by Django 3.1.5 on 2021-01-22 20:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0003_auto_20210121_2224'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasetmodel',
            name='img_list',
            field=models.ManyToManyField(to='datasets.ImageModel'),
        ),
    ]