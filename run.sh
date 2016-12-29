#source activate tensorflow

# Default setting (Default_STD = 0.1)
python mnist.py --logdir=tmp/default

# FanIn product = 1
python mnist.py --logdir=tmp/fanin1 --fan_in_product=1

# FanIn product = 2
python mnist.py --logdir=tmp/fanin2 --fan_in_product=2

# FanOut product = 1
python mnist.py --logdir=tmp/fanout1 --fan_out_product=1

# FanOut product = 2
python mnist.py --logdir=tmp/fanout2 --fan_out_product=2

tensorboard --logdir=tmp
