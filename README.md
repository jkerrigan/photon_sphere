# Photon Sphere 
[![Build Status](https://travis-ci.org/jkerrigan/photon_sphere.svg?branch=master)](https://travis-ci.org/jkerrigan/photon_sphere)[![Coverage Status](https://coveralls.io/repos/github/jkerrigan/photon_sphere/badge.svg?branch=master)](https://coveralls.io/github/jkerrigan/photon_sphere?branch=master)

<p align="center">
<a href="url"><img src="https://github.com/jkerrigan/photon_sphere/blob/master/images/messier_87.jpg" align="center" height="256" width="256"></a>
</p>


Photon Sphere aims to provide a machine learning approach to identifying domain DNS requests that are seen as pernicious (analytics, trackers, ad-serving) for use along with Pi Hole (https://pi-hole.net/) while being deployable on a Raspberry Pi. Model uses the unsupervised text tokenizer YouTokenToMe to parse and tokenize domains for use in a lightweight embedding model. Ideally, common elements (e.g. domain names having words such as 'ads' or 'tracker') among prior known pernicious domains can be used to identify domains that would traditionally require parsing by hand or an exceptionally complicated regex.

The model is composed of a siamese embedding layer with a distance metric learning network. The model is trained using a triplet loss to maximize dissimilarites between domains (e.g. login.microsoft.com - analytics.microsoft.com) while minimizing similarities (e.g. login.github.com - github.com).

## Notes
- YouTokenToMe(YTTM) vocab size is 300 by default (too large results in overfitting)
- Model can be run in real-time or on the archived Pi Hole SQL DNS query logs
- 

## Requirements
- tensorflow
- numpy
- sqlalchemy
- [youtokentome](https://github.com/VKCOM/YouTokenToMe)
