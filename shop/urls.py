from . import views
from django.urls import path

urlpatterns = [
    path("",views.index,name="shophome"),
    path("shop/",views.index,name="shophome"),
    path("register/",views.custom_register,name="register"),
    path("login/",views.custom_login,name="login"),
    path("logout/",views.custom_logout,name="logout"),
    path("shop/checkout/",views.checkout,name="checkout"),
    path("shop/tracker/",views.tracker,name="trackingstatus"),
    path("shop/products/<int:myid>",views.prodview,name="viewingproduct"),
]