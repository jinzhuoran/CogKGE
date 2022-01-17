from utils.similarity_utils import *

from models import *

log_timeout = 1


class Insert(object):
        entity_data = params
        entity_obj = Entity.objects.create(**entity_data)
