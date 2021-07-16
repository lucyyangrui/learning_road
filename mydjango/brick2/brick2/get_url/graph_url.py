from django.conf.urls import url
from brick2.get_view.graph_view import GraphClassifyViewSet

graph_MANAGE = GraphClassifyViewSet.as_view({
    'get': 'get'
})

urlpatterns = [
    url(r'^graph$', graph_MANAGE)
]