from django.contrib import admin

# Register your models here.
from .models import Product,Order,OrderUpdate,UserProductFreq

class Productadmin(admin.ModelAdmin):
    list_display=('product_id','product_display_name', 'gender', 'master_category','sub_category','article_type','base_color','season','year','usage','price')

class Orderadmin(admin.ModelAdmin):
    list_display=('order_id','items_json','amount','user','address')


class Orderupdateadmin(admin.ModelAdmin):
    list_display=('update_id','order_id','update_desc','timestamp')

class Userprofreqadmin(admin.ModelAdmin):
    list_display=('user','prod','freq')

admin.site.register(Product,Productadmin)
admin.site.register(Order,Orderadmin)
admin.site.register(OrderUpdate,Orderupdateadmin)
admin.site.register(UserProductFreq,Userprofreqadmin)


