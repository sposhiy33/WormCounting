from .p2pnet import build_p2p, build_multiclass
from .linear import build_linear
# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if args.linear:
        return build_linear(args, training)
    else:
        return build_p2p(args, training)


def build_classifier(args, training=False):
    return build_multiclass(args, training)
