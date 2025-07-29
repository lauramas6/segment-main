Included Architectures:

SegFormer

Mask2Former

SETR

Your dataset folder must include the following folders containing your rgb images and segmentation masks:

train

train_labels

val

val_labels

test

test_labels

It also must include a class_dict.csv which can be formatted as such: 

name,r,g,b

Sunlit Leaves,0,255,0

Noise,165,42,42

Optional CLI arguments for running train.py and evaluate.py are as follows:

--architecture (segformer,mask2former,setr....)

--data_root (your dataset folder name - i.e. tomato)

For evaluate.py only:

--weights (path to your saved .pt file)

Example of how to run train.py:

python3 train.py --data_root tomato/segformer --architecture segformer

There is a util script named format_dataset.py which can be run on a dataset folder to format it for a particular model architecture. Here is an example of that code: 

python3 utils/format_dataset.py --data_root tomato --architecture setr
