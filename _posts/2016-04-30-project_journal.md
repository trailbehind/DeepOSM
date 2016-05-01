---
title: Project Journal - April 30, 2016
category: notes
layout: post
---

Making progress on a lot of fronts.

## Experimental Results Table

I started compiling a [table of experimental results](https://gaiagps.quip.com/TjKoAJHin8MT), which seems better than bullets in blog posts... maybe this should be auto-logged to a CSV though.

## Tennis Courts

I added some code so that the analysis can target tennis courts, instead of roads - that was easy, and a preliminary analysis is running now.

## CIFAR Classifier

I got a classifier for CIFAR running, which seems suited to RGB images (hopefully RGB-IR). I hope this breaks me out of the 70% range on overnight runs. It may take a GPU to get there though, which should thankfully arrive this coming week.

The MNIST classifier chokes on more than 2 bands, but that might be because I'm not normalizing the RGB number between -1 through 1, or some other mistake in the NN architecture.

## Jupyter Notebooks

@silberman taught me to use jupyter notebooks, which seems pretty useful.


