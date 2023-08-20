import datetime
from collections import deque
from django.shortcuts import render,redirect,HttpResponse
from .models import Product,UserProductFreq,OrderUpdate,Order
import math
import json
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import pathlib 
import os
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

import pandas as pd 
 
import joblib
 
 
import cv2
 
from sklearn.metrics.pairwise import cosine_similarity
 
from pandas.core.common import flatten
 
 

import torch
import torch.nn as nn
 
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
 
from PIL import Image
 
import swifter
import warnings
 
import os

initial_datetime = datetime.datetime.now().timestamp()

 
 
DATASET_PATH= os.getcwd() + "\\media\\"
 
 
df = pd.read_csv(DATASET_PATH + 'styles.csv', on_bad_lines='skip')
 
def get_all_filenames(directory):
    """
    a list of all filenames in the specified directory is returned.
    """
    filenames = {entry.name for entry in os.scandir(directory) if entry.is_file()}
    return filenames
 
images = get_all_filenames(DATASET_PATH + "images/")
 
def check_image_exists(image_filename):
    """
    determines if the requested filename is present among the filenames in the specified directory. Returns False if the file doesn't exist; true if it does.
    """
    global images
    if image_filename in images:
        return image_filename
    else:
        return np.nan
 
df['image'] = df["id"].apply(lambda image: check_image_exists(str(image) + ".jpg"))
df = df.reset_index(drop=True)
 
def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    """Plot a dictionary of figures.
 
    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
 
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    
#image path 
def image_location(img):
    return DATASET_PATH + '\\images\\'  + img
 
# function to load image
def import_img(image):
    image = cv2.imread(image_location(image))
    return image
 
#ResNet18 PyTorch model to convert

'''
As the name implies, we will employ the current architecture, which has 18 layers overall and levels of convolution.
It has the ability to categorize over 1000 different types of objects because it was trained on millions of photos from the ImageNet dataset.
the input shape being defined
'''

 
width= 224
height= 224
 
# Loading the pretrained model
resnetmodel = models.resnet18(pretrained=True)
 
# Use the model object to select the desired layer
layer = resnetmodel._modules.get('avgpool')
 
resnetmodel.eval()
 
# scaling the data
s_data = transforms.Resize((224, 224))
 
#normalizing
standardize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
 
# converting to tensor
convert_tensor = transforms.ToTensor()
 
#missing image object
missing_img = []
#function to get embeddings
 
def vector_extraction(resnetmodel, image_id):
    
    # Using concept of exception handling to ignore missing images
    try: 
        # 1. Load the image with Pillow library
        img = Image.open(image_location(image_id)).convert('RGB')
        
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
        
        # 3. Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        embeddings = torch.zeros(512)
        
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))
            
        # 5. Attach that function to our selected layer
        hlayer = layer.register_forward_hook(copy_data)
        
        # 6. Run the model on our transformed image
        resnetmodel(t_img)
        
        # 7. Detach our copy function from the layer
        hlayer.remove()
        emb = embeddings
        
        # 8. Return the feature vector
        return embeddings
    
    # If file not found
    except FileNotFoundError:
        # Keep track of the index for such items in the missing_img list, then delete them.
        missed_img = df[df['image']==image_id].index
        # print(missed_img)
        missing_img.append(missed_img)
 
# importing the embeddings  
df_embs = pd.read_csv(DATASET_PATH + 'df_embs.csv')
df_embs.drop(['Unnamed: 0'],axis=1,inplace=True)
df_embs.dropna(inplace=True)
 
#exporting as pkl
joblib.dump(df_embs, DATASET_PATH + 'df_embs.pkl', 9)
 
#importing the pkl
df_embs = joblib.load(DATASET_PATH + 'df_embs.pkl')
 
# Comparing photos for similarity (using embedding values)
cosine_sim = cosine_similarity(df_embs) 
 
# just checking the structure of cosine_sim by previewing the first four rows and four columns of similarity
cosine_sim[:4, :4]
 
# Storing the index values in a series index_vales for recommending
index_vales = pd.Series(range(len(df)), index=df.index)
 
 
# Defining a function that gives recommendations based on the cosine similarity score
def recomend_images(ImId, top_n = 6):
    
    # Assigning index of reference into sim_ImId
    sim_ImId    = index_vales[ImId]
    
    # Storing cosine similarity of all other items with item requested by user in sim_scores as a list
    sim_scores = list(enumerate(cosine_sim[sim_ImId]))
    
    # Sorting the list of sim_scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Extracting the top n values from sim_scores
    sim_scores = sim_scores[1:top_n+1]
    
    # ImId_rec will return the index of similar items
    ImId_rec    = [i[0] for i in sim_scores]
    
    # ImId_sim will return the value of similarity score
    ImId_sim    = [i[1] for i in sim_scores]
    
    return index_vales.iloc[ImId_rec].index, ImId_sim
 
#function to get embeddings
def recm_user_input(image_id):
    
    # Exception to handle missing images 
    # print("\n\n\n\nImage ID: ", image_id, "\n\n\n\n")
    img = Image.open(DATASET_PATH + "images\\" +image_id).convert('RGB')
    # print(type(img))
        
    t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
       
    embeddings = torch.zeros(512)
        #print('H',embeddings)
    def select_d(m, i, o):
        embeddings.copy_(o.data.reshape(o.data.size(1)))
    hlayer = layer.register_forward_hook(select_d)
    resnetmodel(t_img)
    hlayer.remove()
    emb = embeddings
    
    cs = cosine_similarity(emb.unsqueeze(0),df_embs)
    cs_list = list(flatten(cs))
    cs_df = pd.DataFrame(cs_list,columns=['Score'])
    cs_df = cs_df.sort_values(by=['Score'],ascending=False)
        
# Printing Cosine Similarity
    # Extracting the index of top 10 similar items/images
    top10 = cs_df[:10].index
    top10 = list(flatten(top10))
    images_list = []
    for i in top10:
        image_id = df[df.index==i]['image']
        # print(image_id)
        images_list.append(image_id)
    images_list = list(flatten(images_list))    
# Generating a dictionary { index, image }
    figures = {'im'+str(i): Image.open(DATASET_PATH + "\\images\\"  + i) for i in images_list}
    fig, axes = plt.subplots(2, 5, figsize = (8,8) )
    for index,name in enumerate(figures):
        axes.ravel()[index].imshow(figures[name])
        axes.ravel()[index].set_title(name)
        axes.ravel()[index].set_axis_off()
    plt.tight_layout()
 
        #return embeddings   
    figures = {'im'+str(i):import_img(row.image) for i, row in df.sample(6).iterrows()}
    return images_list
 
 
 
# Create your views here.
@login_required
def index(request):
    allProds = []
    prev_products=UserProductFreq.objects.filter(user=request.user).order_by('-freq')
    recommend_list=deque([])
    for pp in prev_products:
        recommend_list.extend(recm_user_input(f'{pp.prod}.jpg'))
    recomm_prod_list=[]
    for p in recommend_list:
        pid=p[:-4]
        recomm_prod_list.append(Product.objects.get(product_id=pid))
    if recomm_prod_list: recomm_prod_list.pop(0)
    n=len(recomm_prod_list)
    nslides = n // 4 - math.ceil((n / 4) - (n // 4))
    allProds.append([recomm_prod_list, range(1, nslides), nslides, 'Recommended Products'])

    user=request.user
    catprods = Product.objects.values('master_category', 'product_id')
    cats = {item['master_category'] for item in catprods}
 
    for cat in cats:
        prod = Product.objects.filter(master_category=cat)[:25]
        n = len(prod)
        nslides = n // 4 - math.ceil((n / 4) - (n // 4))
        allProds.append([prod, range(1, nslides), nslides, cat])
    params = {'allProds': allProds,'user':user}
    return render(request, 'shop/index.html', params)
 

def custom_register(request):
    if request.method=='POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')
        if username != '' and password != '' and password == password2 and email != '':
            if User.objects.filter(email=email).exists():
                messages.warning(request, 'Email already in use')
                return redirect('register')
            elif User.objects.filter(username=username).exists():
                messages.warning(request, 'Username already in use')
                return redirect('register')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                userlogin = authenticate(username=username, password=password)
                login(request, userlogin)
                messages.info(request, 'You have been successfully signed up!')
                return redirect('shophome')
        elif username == '':
            messages.warning(request, 'Enter the username')
            return redirect('register')
        elif email == '':
            messages.warning(request, 'Enter the email ID')
            return redirect('register')
        elif password == '' or password != password2:
            messages.warning(request, 'Confirmation password did not match with the given password')
            return redirect('register')
    else:
        params={}
        return render(request, 'shop/register.html',params)

def custom_login(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
        user=authenticate(username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('shophome')
        else:
            messages.warning(request,'Invalid Credentials. Please put down correct username and password')
            return redirect('login')
    return render(request,'shop/login.html')

@login_required
def custom_logout(request):
    logout(request)
    return redirect('login')



def tracker(request):
    if request.method == 'POST':
        orderid = request.POST.get('orderid', '')
        email = request.POST.get('email', '')
        try:
            order = Order.objects.filter(order_id=orderid, email=email)
            if len(order) > 0:
                update = OrderUpdate.objects.filter(order_id=orderid)
                updates = []
                for item in update:
                    strdate = str(item.timestamp)
                    date = datetime.date(int(strdate[:4]), int(strdate[5:7]), int(strdate[8:]))
                    curtime = date.ctime()
                    updates.append(
                        {'text': item.update_desc, 'time': curtime[:3] + ', ' + curtime[4:10] + ' ' + curtime[-4:]})
                    response = json.dumps({'status':'success','updates':updates[::-1],'itemsjson': order[0].items_json}, default=str)
                return HttpResponse(response)
            else:
                return HttpResponse('{"status":"no item"}')
        except Exception as e:
            return HttpResponse('{"status":"error"}')
    return render(request, 'shop/tracker.html')


def prodview(request, myid):
    product = Product.objects.get(product_id=myid)
    user=request.user
    upf=UserProductFreq.objects.get_or_create(user=user, prod=product)
    upf[0].freq+=1*(int(datetime.datetime.now().timestamp()) - initial_datetime)
    upf[0].save()
    allProds = []
    recommend_list=deque([])
    recommend_list.extend(recm_user_input(f'{myid}.jpg'))
    print(recommend_list)
    recomm_prod_list=[]
    for p in recommend_list:
        pid=p[:-4]
        recomm_prod_list.append(Product.objects.get(product_id=pid))
    if recomm_prod_list: recomm_prod_list.pop(0)
    n=len(recomm_prod_list)
    nslides = n // 4 - math.ceil((n / 4) - (n // 4))
    allProds.append([recomm_prod_list, range(1, nslides), nslides, 'Similar Products'])
    return render(request, 'shop/productview.html', {'product': product,'allProds': allProds})


def checkout(request):
    thank = False
    user=request.user
    if request.method == 'POST':
        items_json = request.POST.get('itemsJson', '')
        user=request.user
        amount = request.POST.get('amount', '')
        address = request.POST.get('address', '')
        city = request.POST.get('city', '')
        state = request.POST.get('state', '')
        pin = request.POST.get('pin', '')
        phone = request.POST.get('phone', '')
        order = Order(items_json=items_json, user=user, address=address, phone=phone, city=city, state=state, pin=pin, amount=amount)
        json_object = json.loads(items_json)
        order.save()
        update = OrderUpdate(order_id=order, update_desc='The order has been placed.')
        update.save()
        for prodid in json_object:
            upf=UserProductFreq.objects.get_or_create(user=user, prod=Product.objects.get(product_id=prodid[2:]))
            upf[0].freq+=5 * (int(datetime.datetime.now().timestamp()) - initial_datetime)
            upf[0].save()
        thank = True
        id = order.order_id
        return render(request, 'shop/checkout.html', {'thank': thank, 'id': id})
    return render(request, 'shop/checkout.html')