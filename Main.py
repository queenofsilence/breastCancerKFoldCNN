from usingKFold import Data
def main():
   data = Data()
   X,Y = data.SplitingData()
   data.kfold_Split(10, 10) # first number is split number second one is the number of epochs => kfold_split(split, epoch)
main()