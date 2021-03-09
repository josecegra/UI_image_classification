# Generated by Django 3.1.5 on 2021-01-21 21:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='datasetmodel',
            name='img_list',
        ),
        migrations.AddField(
            model_name='imagemodel',
            name='dataset',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='datasets.datasetmodel'),
        ),
    ]
