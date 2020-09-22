# Photon Sphere 
[![Build Status](https://travis-ci.org/jkerrigan/photon_sphere.svg?branch=master)](https://travis-ci.org/jkerrigan/photon_sphere)[![Coverage Status](https://coveralls.io/repos/github/jkerrigan/photon_sphere/badge.svg?branch=master)](https://coveralls.io/github/jkerrigan/photon_sphere?branch=master)

<a href="url"><img src="https://github.com/jkerrigan/photon_sphere/blob/master/images/messier_87.jpg" align="center" height="256" width="256" ></a>


Photon Sphere attempts to provide a machine learning approach to identifying ad-serving domains. Model consists of the SentencePiece unsupervised tokenizer to parse and tokenize domains for use in a lightweight embedding model. Ideally, common elements (e.g. domains having words such as 'ads' or 'tracker') among prior known domains can be used to identify domains that would traditionally require parsing by hand or an exceptionally complicated regex. The model also takes into account several domains within a 1 sec bucket, to account for the burst like nature of many ad-serving and tracking website dns requests that occur almost simultaneously when a website is accessed.
