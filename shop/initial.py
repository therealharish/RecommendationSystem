import csv 
from .models import Product

with open('/Dataset/styles.csv', mode='r') as file:
    csvFile = csv.reader(file) 

    for line in csvFile:

        image = '/Dataset/images/' + line[0] + '.jpg'
        print(image)

        product = Product(
            product_id=line[0],
            gender=line[1],
            master_category=line[2],
            sub_category=line[3],
            article_type=line[4],
            base_color=line[5],
            season=line[6],
            year=line[7],
            usage=line[8],
            product_display_name=line[9],
            image=image
        )

        product.save()
        break