# Todos
Some notes on todos, didn't want to muck with Github issues for this.

## What's Next
* are major freeways not being rendered?
* add buildings and any other labelable features too, so there is less image that's just "not road"
* FIND - have you seen a paper from a few years ago about estimating osm completeness by comparing size of compressed satellite images vs number of osm nodes
* READ - this presentation on using GPS traces to suggest OSM edits (Strava/Telenav): http://webcache.googleusercontent.com/search?q=cache:VoiCwRHOyLUJ:stateofthemap.us/map-tracing-for-millennials/+&cd=3&hl=en&ct=clnk&gl=us


## Later
* ORDERED - run this on the Linux box with a GPU, see if that's tons faster
* rotate training images (makes the training overfit less)
* classify at the pixel level, by guessing at center pixel roadiness, instead of total tile roadiness
* instead of roads on/off, classify pixels into types (highway, footway, cycleway, lanes-1, lanes-2, lanes-3, etc)
* visualize training on TensorBoard (better than print statements?)
* try recurrent neural nets
* move analysis to the cloud
* label pixels with elevation data too 

## Done
* ~~make config file for all data.analysis constants~~
* ~~use more geographic area for the analysis~~
* ~~print how much loss is falling, rolling average, help with debugging~~
* ~~try just classifying big roads, with 2 or more lanes, or throwing out non motorways (easier as expected)~~
* ~~add another satellite image as validation data~~
* ~~assign more pixels for roads, based on road type (this will turn on more tiles, should be more accurate)~~
* ~~make a Tennis Court list for Dan (classify non-road features like buildings, tennis courts, or baseball diamonds)~~
* ~~use a more suitable (but still off-the-shelf) neural net model (like the cifar10 classifier)~~
* ~~make it work with 4 bands, not just IR band~~
* ~~try balancing RGBs between -1 to 1, instead of 0 to 1~~
