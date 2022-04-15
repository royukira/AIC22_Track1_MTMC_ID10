from .label_smooth import LabelSmoothing
from torch import nn
from .angular_penalty import AngularPenaltySMLoss, ArcMarginProduct
from .circle_loss import CircleLoss
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

is_angular_penalty = False


# TODO: Circle Loss, Triplet Loss等相似性损失函数暂时不支持单独使用
def get_loss(opt, feature_dim=2048) -> dict:
    if opt.triplet and opt.circle:
        raise ValueError("Cannot use triplet loss and circle loss at the same time.")
    loss_dict = {}
    if opt.angular_penalty is not None:
        criterion = AngularPenaltySMLoss(opt.angular_penalty)
        loss_dict = {"criterion": criterion}
        if opt.triplet:
            # Triplet + Angular
            trip_loss = TripletLoss(opt.triplet_margin)
            loss_dict.setdefault("similarity", trip_loss)
            if opt.center:
                # Triplet + Angular + Center
                center_loss = CenterLoss(num_classes=opt.num_labels, 
                                        feat_dim=feature_dim, use_gpu=True)
                loss_dict.setdefault('center', center_loss)
        elif opt.circle:
            # Circle + Angular
            circle_loss = CircleLoss(m=0.25, gamma=64) # gamma = 64 may lead to a better result.
            loss_dict.setdefault("similarity", circle_loss)
            if opt.center:
                # Circle + CE + Center
                center_loss = CenterLoss(num_classes=opt.num_labels, 
                                        feat_dim=feature_dim, use_gpu=True)
                loss_dict.setdefault('center', center_loss)
    else:
        if opt.smooth != 0:
            if opt.triplet:
                # Triplet + CE with Label smoothing
                ce_lm_loss = LabelSmoothing(smoothing=opt.smooth)
                trip_loss = TripletLoss(opt.triplet_margin)
                loss_dict = {"criterion": ce_lm_loss, "similarity": trip_loss}
                if opt.center:
                    # Triplet + CE with Label smoothing + Center
                    center_loss = CenterLoss(num_classes=opt.num_labels, 
                                            feat_dim=feature_dim, use_gpu=True)
                    loss_dict.setdefault('center', center_loss)
            elif opt.circle:
                # Circle + CE with Label smoothing
                ce_lm_loss = LabelSmoothing(smoothing=opt.smooth)
                circle_loss = CircleLoss(m=0.25, gamma=64) # gamma = 64 may lead to a better result.
                loss_dict = {"criterion": ce_lm_loss, "similarity": circle_loss}
                if opt.center:
                    # Circle + CE with Label smoothing + Center
                    center_loss = CenterLoss(num_classes=opt.num_labels, 
                                            feat_dim=feature_dim, use_gpu=True)
                    loss_dict.setdefault('center', center_loss)
            else:
                # CE with Label smoothing
                criterion = LabelSmoothing(smoothing=opt.smooth)
                loss_dict = {"criterion": criterion}
        else:
            if opt.triplet:
                # Triplet + CE
                ce_loss = nn.CrossEntropyLoss()
                trip_loss = TripletLoss(opt.triplet_margin)
                loss_dict = {"criterion": ce_loss, "similarity": trip_loss}
                if opt.center:
                    # Triplet + CE + Center
                    center_loss = CenterLoss(num_classes=opt.num_labels, 
                                            feat_dim=feature_dim, use_gpu=True)
                    loss_dict.setdefault('center', center_loss)
            elif opt.circle:
                # Circle + CE
                ce_loss = nn.CrossEntropyLoss()
                circle_loss = CircleLoss(m=0.25, gamma=64) # gamma = 64 may lead to a better result.
                loss_dict = {"criterion": ce_loss, "similarity": circle_loss}
                if opt.center:
                    # Circle + CE + Center
                    center_loss = CenterLoss(num_classes=opt.num_labels, 
                                            feat_dim=feature_dim, use_gpu=True)
                    loss_dict.setdefault('center', center_loss)
            else:
                # CE
                criterion = nn.CrossEntropyLoss()
                loss_dict = {"criterion": criterion}
    
    return loss_dict
