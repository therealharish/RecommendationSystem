# Generated by Django 3.2.9 on 2022-01-10 19:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0002_auto_20211205_1958'),
    ]

    operations = [
        migrations.CreateModel(
            name='Contact',
            fields=[
                ('msg_id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('mail', models.CharField(default='', max_length=50)),
                ('phno', models.CharField(default='', max_length=20)),
                ('address', models.CharField(default='', max_length=100)),
            ],
        ),
    ]
