import preprocess
import model
import argparse as ae

parser = ae.ArgumentParser()
parser.add_argument("-d","--data",type=str,required=True)
parser.add_argument("-b","--batch_size",type=str,default=32)
args = parser.parse_args()

preprocessor = preprocess.PreProcessor(args.batch_size,args.data)
data = preprocessor.read_file()

print(data)
