Commands for training:

CIFAR10:
    ResNet20: 
        weight:
            # uses l2 regulization as original
            python3 ResNet20_cifar10l2.py w 1 
        activation:
            # uses l2 dropout for better performance.
            # this uses trainer2
            python3 ResNet20_cifar10tr2b.py a 1 
            # or ResNet20_cifar10tr2bHTD.py for biReal structure
        both:
            # need to run activation first, or comment out code for loading weights (starting at line 233)
            # this also uses trainer2
            python3 ResNet20_cifar10tr2b.py b 1 
            # or ResNet20_cifar10tr2bHTD.py
    VGG-Small:
        python3 VGGs_cifar10l2.py w

        # this automaically train full binary afterwards.
        python3 VGGs_cifar10.py a 

        # full binary may also be trained directly.
        python3 VGGs_cifar10.py b 

MNIST:
    # weight
    python3 Dense_mnist.py 1 0 

    # activation
    python3 Dense_mnist.py 0 1 

    # both
    python3 Dense_mnist.py 1 1 

DogCat:
    # First download dataset and place images manually into folders train | validation | test. (last 1300*2 for testing then followed by last 1200*2 for validation)
    # weight
    python3 Conv_DogCat.py w 

    # activation
    python3 Conv_DogCat.py a 

    # both
    python3 Conv_DogCat.py b 

Speech:
    # First download dataset.
    # and save dataset: 
    python3 speech.py

    # weight
    python3 Conv_Speech.py w 

    # activation
    python3 Conv_Speech.py a 

    # both
    python3 Conv_Speech.py b 