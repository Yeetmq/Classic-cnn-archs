import torch
from torch import nn

model_dict = {
    'layer1': [
        {'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'pool', 'pool_type': nn.MaxPool2d, 'kernel_size': 2, 'stride': 2}
    ],
    'layer2': [
        {'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'pool', 'pool_type': nn.MaxPool2d, 'kernel_size': 2, 'stride': 2}
    ],
    'layer3': [
        {'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'pool', 'pool_type': nn.MaxPool2d, 'kernel_size': 2, 'stride': 2}
    ],
    'layer4': [
        {'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'pool', 'pool_type': nn.MaxPool2d, 'kernel_size': 2, 'stride': 2}
    ],
    'layer5': [
        {'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'norm': True, 'activation': nn.ReLU()},
        {'type': 'pool', 'pool_type': nn.MaxPool2d, 'kernel_size': 2, 'stride': 2}
    ],
}

def create_layer(model_cfg: dict) -> nn.Module:
    if model_cfg['type'] == 'conv':
        conv_layer = []
        conv = nn.Conv2d(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            kernel_size=model_cfg['kernel_size'],
            stride=model_cfg['stride'],
            padding=model_cfg['kernel_size'] // 2
        )
        conv_layer.append(conv)
        if model_cfg['norm']:
            batch_normalization = nn.BatchNorm2d(model_cfg['out_channels'])
            conv_layer.append(batch_normalization)
        if model_cfg['activation'] is not None:
            act = model_cfg['activation']
            conv_layer.append(act)
        return nn.Sequential(*conv_layer)
    elif model_cfg['type'] == 'pool':
        pool_layer = model_cfg['pool_type'](kernel_size=model_cfg['kernel_size'], stride=model_cfg['stride'])
        return pool_layer

def create_extractor(model_cfg: dict):
    layers_dict = {}
    out_channels_list = []
    out_strides = []
    for layer_name, sublayer in model_cfg.items():
        layers_list = []
        for subsublayer in sublayer:
            layers_list.append(create_layer(subsublayer))
            if 'out_channels' in subsublayer:
                out_channels = subsublayer['out_channels']
                out_channels_list.append(out_channels)
            if 'stride' in subsublayer:
                stride = subsublayer['stride']
                out_strides.append(stride)
        layers_dict[layer_name] = nn.Sequential(*layers_list)
    return layers_dict, out_channels_list, out_strides

class VGG(nn.Module):
    def __init__(self, model_cfg: dict=model_dict, dropout: float=0.5, num_classes: int=42):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        layers_dict, out_channels_list, out_strides = create_extractor(model_cfg)
        self.out_channels = out_channels_list
        self.output_stride = out_strides
        self.extractor = nn.ModuleDict(layers_dict)

    def forward(self, x):
        for _, layer in self.extractor.items():
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
