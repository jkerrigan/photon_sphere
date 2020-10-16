# Photon Sphere 
[![Build Status](https://travis-ci.org/jkerrigan/photon_sphere.svg?branch=master)](https://travis-ci.org/jkerrigan/photon_sphere)[![Coverage Status](https://coveralls.io/repos/github/jkerrigan/photon_sphere/badge.svg?branch=master)](https://coveralls.io/github/jkerrigan/photon_sphere?branch=master)

<p align="center">
<a href="url"><img src="https://github.com/jkerrigan/photon_sphere/blob/master/images/messier_87.jpg" align="center" height="256" width="256"></a>
</p>


Photon Sphere attempts to provide a machine learning approach to identifying ad-serving domains for use along with Pi-Hole and is entirely deployable on a Raspberry Pi. Model uses the unsupervised text tokenizer YouTokenToMe to parse and tokenize domains for use in a lightweight embedding model. Ideally, common elements (e.g. domain names having words such as 'ads' or 'tracker') among prior known domains (training dataset) can be used to identify domains that would traditionally require parsing by hand or an exceptionally complicated regex. The model uses a contextual domain request approach to identifying adware related DNS requests by bucketing DNS requests according to some time cadence (1 sec by default). This allows for the model to account for the burst like nature of many ad-serving and tracking website DNS requests that occur almost simultaneously when a website is accessed.

## Notes
- ~~bucketing cadence has an initial buffer size of 100 DNS requests (see below for example)~~
- YouTokenToMe(YTTM) vocab size is 300 by default (too large results in overfitting)
- Model can be run actively or on the archived Pi-Hole SQL DNS query logs

`[domain1.adserv.com domain2.adserv.com ... domain100.adserv.com]` <- 1 sec bucket

## Requirements
- tensorflow
- numpy
- sqlalchemy
- [youtokentome](https://github.com/VKCOM/YouTokenToMe)
