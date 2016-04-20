---
title: Project Journal - April 19, 2016
category: notes
layout: post
---

## Overall Notes for the Day

I should put these bullets into a spreadsheet going forward.

## Data Science

* Ran 5000 training batches, patch size 5. Still 51% on training data, guessing 75% were on this time. Loss got down to 18, as compared to 28 for same run with patch size 6

* Switched to tile size 12 to get more data and see how that worked. Keeping other stats from last run. Loss remains in the 50s entire run. Probably because most labels are OFF in this training set, I could even them out manually actually.

* Retrying with tile size 28. My code can only handle divisible by 4 tile sizes I think. Loss only drops to low 60s in this case. 47.5% accurate, 65.7% guess true.

* Trying a new 5000 batch test with size 12px images, 6 patch, but this time enforcing an equal number of training/test labels. 59% accurate, 42.4% of predictions guess True, loss dropped to mid 60s. (25000 labeled images) test accuracy 55%, 76.5% of predictions guess True, [SHUFFLING BUG PRESENT, MOST TEST DATA FROM NORTH]

* Trying a 3X (15,000) batch test with size 12px images, 6 patch.  [SHUFFLING BUG PRESENT, MOST TEST DATA FROM NORTH]

* Going back to 5000 batch, same parameters otherwise, but fixed the shuffling bug. [STILL BUG, FIXED IT]

 * Same, but had another bug where I was setting the labels with one bit of code, and evaling the predictions with another. FIXED THAT TOO NOW.

 * Same with randomization bugs fixed, output looks right, but 12px tile predictions still suck, loss only falls to 55, predictions are 88% on (too much).

 * Trying new 48px tiles, 5000 batches, same patch size of 5.

## Programming

 * added a way to generate equal on/off training sets from any size tile, to facilitate using smaller tiles

