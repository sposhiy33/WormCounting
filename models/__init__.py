from .p2pnet import build_p2p
from .classification import build_classifier
from .mlp import build_mlp
from .gat import build_gat_classifier

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if args.classifier:
        return build_classifier(args, training)
    elif args.mlp_classifier or args.mlp:
        return build_mlp(args, training)
    elif args.gat:
        return build_gat_classifier(args, training) 
    else:
        return build_p2p(args, training)

