#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 02:29:59 2020

@author: rafail
"""

import numpy as np 

class OUActionNoise(object):

    def generate_noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


