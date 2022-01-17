from mongoengine import connect

from models import Entity

connect('cogkge', host='210.75.240.136', username='cipzhao2022', password='cipzhao2022', port=1234, connect=False)


def insert_entity(entity):
    # entity = {'name': 'sss', 'description': 'sss', 'type': 'sss'}
    Entity.objects.create(**entity)


def search_entity(keyword='ss'):
    entities = Entity.objects(name__contains=keyword)
    return entities


search_entity()
print(1)
