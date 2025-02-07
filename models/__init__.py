from .p2pnet import build_p2p
from .classification import build_classifier
# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if args.classifier:
        return build_classifier(args, training)
    else:
        return build_p2p(args, training)

