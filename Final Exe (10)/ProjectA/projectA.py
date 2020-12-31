#   Dataset Link:
#   https://drive.google.com/drive/folders/1J4MQNwbOTz_YOWvMeVIubeXqo54BBO1R
#   Dataset Info:
#   http://kdd.ics.uci.edu/databases/kddcup99/task.html
import pandas as pd

train_set = pd.read_csv("Data/NSL-KDDTrain.csv",
                        sep=',',
                        header=1,  # no header, alternative header = header_col
                        index_col=None,  # no index, alternative header = index_row
                        skiprows=0  # how many rows to skip / not include in read_csv
                        )
print(len(train_set))
print(train_set.head(5))

test_set = pd.read_csv("Data/NSL-KDDTest.csv",
                       sep=',',
                       header=1,  # no header, alternative header = header_col
                       index_col=None,  # no index, alternative header = index_row
                       skiprows=0  # how many rows to skip / not include in read_csv
                       )
print(len(test_set))
print(test_set.head(5))
