# Generated by Django 3.1.5 on 2021-01-21 22:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0002_auto_20210121_2123'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imagemodel',
            old_name='annotation',
            new_name='label',
        ),
        migrations.RemoveField(
            model_name='imagemodel',
            name='dataset',
        ),
    ]