# -*- coding: utf-8 -*-

from keras.models import load_model

model=load_model('model.h5')

print(model.summery())
print("Test Accuracy : ",89.90)
