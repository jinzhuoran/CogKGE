import datetime

from mongoengine import StringField, IntField, FloatField, BooleanField, DateTimeField, Document
from mongoengine import connect

connect('cogkge', host='210.75.240.136', username='cipzhao2022', password='cipzhao2022', port=1234, connect=False)


def to_dict_helper(obj):
    return_data = []
    for field_name in obj._fields:
        if field_name in ("id",):
            continue
        data = obj._data[field_name]
        if isinstance(obj._fields[field_name], StringField):
            return_data.append((field_name, str(data)))
        elif isinstance(obj._fields[field_name], FloatField):
            return_data.append((field_name, float(data)))
        elif isinstance(obj._fields[field_name], IntField):
            return_data.append((field_name, int(data)))
        elif isinstance(obj._fields[field_name], BooleanField):
            return_data.append((field_name, bool(data)))
        elif isinstance(obj._fields[field_name], DateTimeField):
            return_data.append(field_name, datetime.datetime.strptime(data))
        else:
            return_data.append((field_name, data))
    return dict(return_data)


class Entity(Document):
    name = StringField(required=True)
    description = StringField(required=True)
    type = StringField(required=True)
    time = StringField(default="2000")

    def to_dict(self):
        return to_dict_helper(self)


def insert_entity(entity):
    # entity = {'name': 'sss', 'description': 'sss', 'type': 'sss'}
    Entity.objects.create(**entity)


def search_entity(keyword):
    entities = Entity.objects(name__contains=keyword)
    return entities

if __name__ == '__main__':
    entity = {'name': 'sss', 'description': 'sss', 'type': 'sss'}
    insert_entity(entity)
    entity = {'name': 'ss', 'description': 'ss', 'type': 'ss'}
    insert_entity(entity)
    for item in search_entity('s'):
        print(item.to_dict())
    # print(search_entity('s'))
    Entity.objects(name='ss').delete()
    print("end")
