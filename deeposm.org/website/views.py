"""Views for deeposm.org."""

from django.http import HttpResponse, JsonResponse
from django.template import loader
import boto3
import json
import operator
import os
import pickle
from website import settings

FINDINGS_S3_BUCKET = 'deeposm'

STATE_NAMES_TO_ABBREVS = {
    'delaware': 'de',
    'maine': 'me',
    'new-hampshire': 'nh',  # nh is unused
}


def home(request):
    """The home page for deeposm.org."""
    template = loader.get_template('home.html')
    return HttpResponse(template.render(request))


def view_error(request, analysis_type, country_abbrev, state_name, error_id):
    """View the error with the given error_id."""
    cache_findings()
    template = loader.get_template('view_error.html')
    errors = sorted_findings(state_name)
    error = errors[int(error_id)]
    context = {
        'error_id': error_id,
        'center': ((error[4] + error[2]) / 2, (error[3] + error[1]) / 2),
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
    errors = sorted_findings(state_name)
    context = {
        'country_abbrev': country_abbrev,
        'state_name': state_name,
        'analysis_type': analysis_type,
        'analysis_title': analysis_type.replace('-', ' ').title(),
        'errors': errors,
    }

    if request.GET.get("json"):
        return JsonResponse(context)

    return HttpResponse(template.render(context, request))


def sorted_findings(state_name):
    """Return a list of errors for the path, sorted by probability."""
    path = 'website/static/' + STATE_NAMES_TO_ABBREVS[state_name] + '/findings.pickle'
    with open(path, 'r') as infile:
        errors = pickle.load(infile)
    return errors


def cache_findings():
    """Download findings from S3."""
    s3 = boto3.resource('s3')
    deeposm_bucket = s3.Bucket(FINDINGS_S3_BUCKET)
    for obj in deeposm_bucket.objects.all():
        local_path = 'website/static/' + obj.key
        try:
            os.mkdir('website/static/' + obj.key.split('/')[0])
        except:
            pass
        if not os.path.exists(local_path):
            s3_client = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
            s3_client.download_file(FINDINGS_S3_BUCKET, obj.key, local_path)
            with open(local_path, 'r') as infile:
                errors = pickle.load(infile)[0:1000]
            errors.sort(key=operator.itemgetter(0), reverse=True)
            with open(local_path, 'w') as infile:
                pickle.dump(errors, infile)

            print("DOWNLOADED {}".format(obj.key))
        else:
            print("ALREADY DOWNLOADED {}".format(obj.key))
