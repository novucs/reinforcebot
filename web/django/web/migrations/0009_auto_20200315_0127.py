# Generated by Django 3.0.3 on 2020-03-15 01:27

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('web', '0008_add_trigram_extension'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='contributor',
            unique_together={('user_id', 'agent_id')},
        ),
    ]
