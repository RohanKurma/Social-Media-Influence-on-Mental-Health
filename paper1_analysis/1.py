# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 05:03:37 2025

@author: rohan
"""

import pandas as pd
import numpy as np

df = pd.read_excel(r"E:\project_dv\paper1\mmc2.xlsx")


social_media = df[df.columns[2]]

social_media.unique()


for i,col in enumerate(df.columns):
    print('*****************************************************************')
    print(f'{i} : {col}')
