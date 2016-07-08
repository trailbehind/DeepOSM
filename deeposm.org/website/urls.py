"""The deeposm.org URL Configuration."""

from django.conf.urls import url
from django.contrib import admin
from website.views import home, list_errors, view_error, refresh_findings


urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^refresh_findings/', refresh_findings),
    url(r'^(?P<analysis_type>[^/]*)/(?P<country_abbrev>[^/]*)/(?P<state_name>[^/]*)/view/'
        r'(?P<error_id>[0-9]+)/$', view_error),
    url(r'^(?P<analysis_type>[^/]*)/(?P<country_abbrev>[^/]*)/(?P<state_name>[^/]*)/list/$',
        list_errors),
    url(r'^', home),
]
