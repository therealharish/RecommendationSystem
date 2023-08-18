from django.db import models
from django.contrib.auth.models import User
# Create your models here.
CATEGORY=(
    (11,'Electronic'),
    (12,'Clothing'),
    (13,'Crockery'),
    (14,'Toys'),
    (15,'Grocery'),
    (16,'Furniture'),
    (17,'Jewellery'),
)


class Product(models.Model):
    product_id=models.BigIntegerField(primary_key=True)
    gender = models.CharField(max_length=10, blank=True)
    master_category = models.CharField(max_length=50, blank=True) 
    sub_category = models.CharField(max_length=50, blank=True) 
    article_type = models.CharField(max_length=50, blank=True) 
    base_color = models.CharField(max_length=50, blank=True)
    season = models.CharField(max_length=50, blank=True) 
    year = models.IntegerField(default=0, blank=True) 
    usage = models.CharField(max_length=50, blank=True) 
    product_display_name = models.TextField(blank=True)
    price=models.IntegerField(default=1000)

    # product_name=models.CharField(max_length=500,blank=False)
    # category=models.PositiveIntegerField(choices=CATEGORY,blank=False)
    # category=models.PositiveIntegerField(blank=False)
    # price=models.IntegerField(default=0)
    # desc=models.TextField(blank=False)
    # publish_date=models.DateField()
    # image=models.ImageField(upload_to="shop/images",default="")

    def __str__(self):
        return f'{self.product_id}'


# class Contact(models.Model):
#     msg_id=models.AutoField(primary_key=True)
#     name=models.CharField(max_length=50)
#     mail=models.CharField(max_length=50,default="")
#     phno=models.CharField(max_length=20,default="")
#     address=models.CharField(max_length=100,default="")
#
#     def __str__(self):
#         return self.name
#
class Order(models.Model):
    order_id=models.AutoField(primary_key=True)
    items_json=models.CharField(max_length=50000)
    amount=models.IntegerField(default=0)
    user=models.ForeignKey(User, on_delete=models.CASCADE)
    address=models.CharField(max_length=100)
    city=models.CharField(max_length=30)
    state=models.CharField(max_length=30,default="")
    pin=models.CharField(max_length=10)
    phone=models.CharField(max_length=10,default="")

    def __str__(self) -> str:
        return f'{self.order_id} {self.items_json} ordered by {self.user}'
    

class OrderUpdate(models.Model):
    update_id=models.AutoField(primary_key=True)
    order_id=models.ForeignKey(Order, on_delete=models.CASCADE)
    update_desc=models.CharField(max_length=5000)
    timestamp=models.DateField(auto_now_add=True)

    def __str__(self):
        return self.update_desc[:80]+'...'
    

class UserProductFreq(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    prod=models.ForeignKey(Product,on_delete=models.CASCADE)
    freq=models.BigIntegerField(default=0)

    def __str__(self) -> str:
        return f'{self.user} ordered {self.prod} - {self.freq} times'

