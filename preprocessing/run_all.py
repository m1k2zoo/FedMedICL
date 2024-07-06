from subprocess import call
import os
from preprocessed_dataset import CheXpert, fitzpatrick17k, HAM10000, OL3I, PAPILA, COVID

os.chdir("preprocessed_dataset")

print("================")
print("Preprocessing CheXpert")
CheXpert.preprocess_CheXpert()
print()

print("================")
print("Preprocessing fitzpatrick17k")
fitzpatrick17k.preprocess_fitzpatrick17k()
print()

print("================")
print("Preprocessing HAM10000")
HAM10000.preprocess_HAM10000()
print()

print("================")
print("Preprocessing OL3I")
OL3I.preprocess_OL3I()
print()

print("================")
print("Preprocessing PAPILA")
PAPILA.preprocess_PAPILA()
print()

print("===============")
print("Preprocessing COVID")
COVID.preprocess_COVID()
