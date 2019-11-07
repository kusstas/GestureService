from __future__ import print_function
import argparse

from torch.autograd import Variable
import torch.onnx
from torch import nn
from torch.nn.init import normal_, constant_


import pretrainedmodels

print("!!! INFO: pretrainedmodels should be modified x=x.view(x.size(0), -1) on x=x.flatten(1) !!!")

class MLPmodule(torch.nn.Module):
    """
    This is the 2-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(MLPmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = 512
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_frames * self.img_feature_dim,
                      self.num_bottleneck),
            nn.Dropout(0.20),  # Add an extra DO if necess.
            nn.ReLU(),
            nn.Linear(self.num_bottleneck, self.num_class),
        )

    def forward(self, input):
        input = self.classifier(input)
        return input


def return_MLP(relation_type, img_feature_dim, num_frames, num_class):
    MLPmodel = MLPmodule(img_feature_dim, num_frames, num_class)
    return MLPmodel



class TSN(nn.Module):
    def __init__(self, num_class, args):
        super(TSN, self).__init__()
        self.modality = args.modality
        self.num_segments = args.num_segments
        self.num_motion = args.num_motion
        self.reshape = True
        self.before_softmax = True
        self.dropout = args.dropout
        self.dataset = args.dataset
        self.crop_num = 1
        self.consensus_type = args.consensus_type # LSTM etc.
        self.img_feature_dim = args.img_feature_dim  # the dimension of the CNN feature to represent each frame
        base_model = args.arch  # resnet etc.
        nhidden = 512
        print_spec = True
        new_length = None
        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = 1

        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, self.consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class, base_model)

        self.consensus = return_MLP(self.consensus_type, self.img_feature_dim, self.num_segments, num_class)

    def _prepare_tsn(self, num_class, base_model):
        print(getattr(self.base_model, self.base_model.last_layer_name))
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        self.base_model = pretrainedmodels.__dict__[base_model](num_classes=1000, pretrained='imagenet')
        self.base_model.last_layer_name = 'last_linear'
        print('last_layer_name', self.base_model.last_layer_name)
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        base_out = self.base_model(input)
        base_out = self.new_fc(base_out)
        return base_out


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

global args

parser = argparse.ArgumentParser(description="test TRN on a single video")
parser.add_argument('--video_file', type=str, default='')
parser.add_argument('--frame_folder', type=str, default='')
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='jester',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default='test')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--dropout', '--do', default=0.3, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--rnn_hidden_size', default=256, type=int, help="rnn hidden laye feature dimension")
parser.add_argument('--rnn_layer', default=1, type=int, help="the number of layers in rnn")
parser.add_argument('--rnn_dropout', default=0.2, type=float,
                    help="the dropout rate applied at rnn layers number of layers in rnn")
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--consensus_type', type=str, default='MLP')
parser.add_argument('--weights', type=str, default='weights.pth.tar')
categories_file = 'category.txt'

args = parser.parse_args()

categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

net = TSN(num_class, args)

checkpoint = torch.load(args.weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()
net.consensus.eval()

dummy_tsn = Variable(torch.ones((8, 3, 224, 224)).cuda())
dummy_mlp = Variable(torch.ones((1, 2048)).cuda())

torch.onnx.export(net,
                              dummy_tsn,
                              "gesture_resnet18_mlp_tsn.onnx",
                              export_params=True,
                              do_constant_folding=True,
                              opset_version=9,
                              verbose=True)

torch.onnx.export(net.consensus,
                              dummy_mlp,
                              "gesture_resnet18_mlp_mlp.onnx",
                              export_params=True,
                              do_constant_folding=True,
                              opset_version=9,
                              verbose=True)
