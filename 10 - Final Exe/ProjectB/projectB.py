#   Dataset Link:
#   https://drive.google.com/drive/folders/1d5NCf33sX4ikTtyG0H8A4I18V-mGIdWm
#   Dataset Info:
#
import pandas as pd

dataset = pd.read_csv("Data/fuel_emissions.csv",
                      sep=',',
                      header=1,  # no header, alternative header = header_col
                      index_col=None,  # no index, alternative header = index_row
                      skiprows=0  # how many rows to skip / not include in read_csv
                      )
print(len(dataset))
print(dataset.head(5))
