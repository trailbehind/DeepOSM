"""Post data to S3."""

import boto3
import os
import pickle

from src.config import CACHE_PATH, FINDINGS_S3_BUCKET
from src.single_layer_network import list_findings
from src.training_data import load_all_training_tiles, tag_with_locations
from src.training_visualization import render_results_for_analysis


def post_findings_to_s3(raster_data_paths, model, training_info, bands, render_results):
    """Aggregate findings from all NAIPs into a pickled list, post to S3."""
    findings = []
    for path in raster_data_paths:
        labels, images = load_all_training_tiles(path, bands)
        if len(labels) == 0 or len(images) == 0:
            print("WARNING, there is a borked naip image file")
            continue
        false_positives, fp_images = list_findings(labels, images, model)
        path_parts = path.split('/')
        filename = path_parts[len(path_parts) - 1]
        print("FINDINGS: {} false pos of {} tiles, from {}".format(
            len(false_positives), len(images), filename))
        if render_results:
            # render JPEGs showing findings
            render_results_for_analysis([path], false_positives, fp_images, training_info['bands'],
                                        training_info['tile_size'])

        # combine findings for all NAIP images analyzedfor the region
        [findings.append(f) for f in tag_with_locations(fp_images, false_positives,
                                                        training_info['tile_size'],
                                                        training_info['naip_state'])]

    # dump combined findings to disk as a pickle
    try:
        os.mkdir(CACHE_PATH + training_info['naip_state'])
    except:
        pass
    naip_path_in_cache_dir = training_info['naip_state'] + '/' + 'findings.pickle'
    local_path = CACHE_PATH + naip_path_in_cache_dir
    with open(local_path, 'w') as outfile:
        pickle.dump(findings, outfile)

    # push pickle to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, FINDINGS_S3_BUCKET, naip_path_in_cache_dir)
