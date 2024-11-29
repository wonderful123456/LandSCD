from models.backbone import resnet, xception, drn, mobilenet, groupXception

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'resnet152':
        return resnet.ResNet152(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'lightxception':
        return xception.LightAlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
        
        
def build_backbone1(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'resnet152':
        return resnet.ResNet152(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXceptionEdge(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError        
        
        
def build_backbone2(backbone, output_stride, GroupNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'resnet152':
        return resnet.ResNet152(output_stride, BatchNorm)
    elif backbone == 'xception':
        return groupXception.AlignedXception(output_stride, GroupNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, GroupNorm)
    else:
        raise NotImplementedError   