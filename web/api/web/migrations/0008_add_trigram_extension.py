from django.contrib.postgres.operations import TrigramExtension
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0007_auto_20200314_2051'),
    ]

    operations = [
        TrigramExtension(),
    ]
