# RecommendationSystem

## This is the user-specified recommendation system project.

### Tech-Stack used:
* Python
Django Framework
* HTML, CSS, JavaScript, and Bootstrap
* ResNet model (machine learning component for recommendation algorithm)

The web application contains sign-up and LogIn facilities.
The products are recommended based on the past activity of the user. The algorithm considers the items added to the cart and purchased at any point of time to
Then, after thorough consideration, the models extract features (which are similar features in the images stored in the data set as well as any match in the name or description).
of other items) and displays all those products presented on the **Recommended Items** carousel.
Each time the user performs a new order, the recommendation data set is updated at the back-end. Hence, the changes are also reflected in the data used by the recommendation algorithm.
This enables the user to view the freshly recommended products each time they perform any transactions.

## Command to create and run this project

### clone the repository

```
git clone https://github.com/therealharish/RecommendationSystem
```

### move to project directory
```
cd RecommendationSystem
```

### install requirements specified in _requirements.txt_
```
pip install -r requirements.txt
```

The data set is provided at: [Images Data set](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?select=images)

### run migration to register models in admin
```
python manage.py makemigrations
python manage.py migrate
```

### run _initial.py_ separately to store all the products from data set to Products model in the SQLite database
```
python initial.py
```

### to run django server on your localhost
```
python manage.py runserver
```

### create a super user for admin panel
```
python manage.py createsuperuser
```





