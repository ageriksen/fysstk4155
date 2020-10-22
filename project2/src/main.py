#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import systems.simplepoly as smpl
import systems.franke as frnk

import numpy as np


def main():
    #rows = 100; cols = 100; sigma = 0.1
    #RunFranke(rows, cols, sigma)
    #franke = frnk.FrankeRegression()
    #franke.RunFranke(rows, cols, sigma
    dtpts = 100
    simple = smpl.SimpleRegression()
    simple.RunSimpleExample(dtpts)


    

    


if __name__ == "__main__":
    main()
