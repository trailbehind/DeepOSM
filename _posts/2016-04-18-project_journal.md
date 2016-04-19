---
title: Project Journal - April 18, 2016
category: notes
layout: post
---

I decided to start journaling because I have a bit of traction, and the analysis isn't totally wrong. Also Lacker said I should have a notebook, but I never write stuff with pens.

## Data Science

### Experiments where number of pixels on tiles must be greater than length of tile * .75

* Ran 5000 training batches last night over night, noticed the data wasn't being properly randomized. Patch size 6.

* Fixed it, ran a new 50 batches. Not very accurate at that level of training, 50/50. Patch size 6.

* Ran a new test of 500 batches. 52% accurate.  Patch size 6.

* Ran a new test of 500 batches with a bigger patch (8px patch instead of 6px). Only 48% accurate, and only predicted 8.5% of test set on.

* Ran a new test of 500 batches with a bigger patch (7px patch). Only 50% accurate, 50% of labels on.

* Started a new test of 5000 batches over night, with a patch size of 5. Again, only 50% accurate.


## Programming

 * fix a bug in data randomization
 * added todos list 
 * added project journal using github pages 
