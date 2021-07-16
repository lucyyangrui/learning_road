from rest_framework import viewsets, response
from rest_framework.response import Response
from brick2.get_models import graph_model
from django.http import HttpResponse


class GraphClassifyViewSet(viewsets.ViewSet):
    def get(self, request):
        res = graph_model.test()
        return HttpResponse(res)
