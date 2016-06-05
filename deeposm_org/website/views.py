"""Views for deeposm.org."""

from django.http import HttpResponse
from django.template import loader
import boto3
import json
import os
import pickle
import settings

TEST_ERROR_DICT = {'id': 1, 'certainty': .6, 'source_image': 'http://foo/some_naip.tiff',
                   'source_image_x': 2, 'source_image_y': 3, 'tile_size': 64,
                   'neLat': 32, 'neLon': -112, 'swLat': 31, 'swLon': -113}


def home(request):
    """The home page for deeposm.org."""
    template = loader.get_template('home.html')
    return HttpResponse(template.render(request))


def view_error(request, analysis_type, error_id):
    """View the error with the given error_id."""
    cache_findings()
    template = loader.get_template('view_error.html')
    with open('website/static/findings.pickle', 'r') as infile:
        error = pickle.load(infile)[int(error_id)]
    context = {
        'error_id': error_id,
        'error': error,
        'json_error': json.dumps(error),
        'analysis_title': analysis_type.replace('-', ' ').title(),
        'analysis_type': analysis_type,
    }
    return HttpResponse(template.render(context, request))


def list_errors(request, analysis_type, country_abbrev, state_name):
    """List all the errors of a given type in the country/state."""
    cache_findings()
    template = loader.get_template('list_errors.html')
    with open('website/static/findings.pickle', 'r') as infile:
        errors = pickle.load(infile)
    context = {
        'country_abbrev': country_abbrev,
        'state_name': state_name,
        'analysis_type': analysis_type,
        'analysis_title': analysis_type.replace('-', ' ').title(),
        'errors': errors,
    }
    return HttpResponse(template.render(context, request))


def cache_findings():
    """Download findings from S3."""
    if not os.path.exists('website/static/findings.pickle'):
        s3_client = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
        s3_client.download_file('deeposm', 'findings.pickle', 'website/static/findings.pickle')
