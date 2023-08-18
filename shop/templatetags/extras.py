from django import template
register=template.Library()

@register.filter(name='get_category')
def get_category(categorydict,key):
    return categorydict.get(key)