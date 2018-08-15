from torch import nn
from torch.nn import Sequential
from .common import conv, act, Concat, norm
from .custom_layers import ConcatTable, fadein_layer, TruncateChannels



# three methods below taken from
# from https://github.com/nashory/pggan-pytorch/blob/master/network.py
def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if name not in names:
            names.append(name)
    return names


def deepcopy_module(module, target):
    new_module = Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)
            new_module[-1].load_state_dict(m.state_dict())
    return new_module


def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0-tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))


# loosely inspired from https://github.com/nashory/pggan-pytorch/blob/master/network.py
class SkipNetwork(nn.Module):
    class Config(object):
        need_sigmoid = True
        need_bias = True,
        pad = 'reflection'
        upsample_mode = 'bilinear'
        downsample_mode = 'stride'
        act_fun = 'LeakyReLU'
        norm_fun = 'BatchNorm'
        need1x1_up = True
        inner_layers = 0
        filter_size_down = 3
        filter_size_up = 3
        filter_size_skip = 1
        input_channels = 32
        output_channels = 3
        # Same number of features and skip connections as for denoising task
        down_channels = [128, 64, 32, 16, 8]  # [8, 16, 32, 64, 128]
        skip_channels = [4,  4,  4,  4,   4]
        # skip_channels = [0] * len(down_channels)

        def __init__(self):
            assert len(self.down_channels) == len(self.skip_channels)
            self.n_channels = len(self.down_channels)

    def __init__(self, **kwargs):
        super().__init__()
        self.config = self.Config()
        for k, v in kwargs.items():
            setattr(self.config, k, v)

        self.level = 0
        self.model = Sequential()

        # We add only from/toRGB blocks with an empty layer in the middle.
        # This network will not be used as is, as it will be grown before the first use.
        self.model.add_module(
            'from_rgb_block',
            self.from_rgb_block(self.get_features_for_level(self.level))
        )
        # self.model.add_module("layer0", Sequential())
        self.model.add_module('to_rgb_block', self.to_rgb_block(self.get_features_for_level(self.level)))

        self.module_names = get_module_names(self.model)

    def forward(self, x):
        return self.model(x)

    def get_features_for_level(self, level):
        """Returns the number of latent features for the given level."""

        if level <= 1:
            level = 1
        if level >= self.config.n_channels:
            level = self.config.n_channels

        return self.config.down_channels[level - 1]

    def get_skip_features_for_level(self, level):
        """Returns the number of skip channels for the given level."""

        if level <= 1:
            level = 1
        if level >= self.config.n_channels:
            level = self.config.n_channels

        return self.config.skip_channels[level - 1]

    def flush(self):
        # print("flushing network")

        high_resl_block = deepcopy_module(self.model.input_concat_block.layer2, 'high_resl_block')
        high_resl_from_rgb = deepcopy_module(self.model.input_concat_block.layer2, 'high_resl_from_rgb')

        # add the high resolution block.
        new_model = Sequential()
        new_model.add_module('from_rgb_block', high_resl_from_rgb)
        new_model.add_module("in_layer_level{}".format(self.level), high_resl_block)

        # add rest.
        for name, module in self.model.named_children():
            if name not in ['input_concat_block', 'output_concat_block',
                            'input_fadein_block', 'output_fadein_block']:
                new_model.add_module(name, module)                      # make new structure and,
                new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

        # make deep copy and paste.
        high_resl_block = deepcopy_module(self.model.output_concat_block.layer2, 'high_resl_block')
        high_resl_to_rgb = deepcopy_module(self.model.output_concat_block.layer2, 'high_resl_to_rgb')

        # now, add the high resolution block.
        new_model.add_module("out_layer_level{}".format(self.level), high_resl_block)
        new_model.add_module('to_rgb_block', high_resl_to_rgb)
        self.model = new_model
        self.module_names = get_module_names(self.model)

    def grow(self):
        # print("growing network")
        self.level = self.level + 1
        new_model = Sequential()

        # Retrieve the fromRGB layer from the previous level
        low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
        prev_fromrgb_block = Sequential()
        prev_fromrgb_block.add_module('low_resl_downsample', nn.AvgPool2d(kernel_size=2))

        prev_fromrgb_block.add_module('low_resl_from_rgb', low_resl_from_rgb)

        # Create the next block made of a new fromRGB layer + a convolutional layer
        next_fromrgb_block = Sequential()
        next_fromrgb_block.add_module(
            'high_resl_from_rgb',
            self.from_rgb_block(
                self.get_features_for_level(self.level)
            ))
        next_fromrgb_block.add_module(
            'high_resl_block',
            self.dec_conv_block(
                self.get_features_for_level(self.level),
                self.get_features_for_level(self.level - 1)
            ))

        new_model = Sequential()

        # If it's the first layer, we don't have to concatenate/fade it in with the previous layer
        new_model.add_module('input_concat_block', ConcatTable(prev_fromrgb_block, next_fromrgb_block))
        new_model.add_module('input_fadein_block', fadein_layer())

        # Add all previous layers to a container
        other_previous_layers = Sequential()
        for name, module in self.model.named_children():
            if 'rgb_block' not in name:
                other_previous_layers.add_module(name, module)
                other_previous_layers[-1].load_state_dict(module.state_dict())

        # Concatenate this container to a skip connection block
        new_model.add_module(
            "skip_concat_block",
            Concat(
                1,
                other_previous_layers,
                self.skip_block(
                    self.get_features_for_level(self.level - 1),
                    self.get_skip_features_for_level(self.level)),
            )
        )

        # Retrieve previous toRGB layer
        low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
        prev_torgb_block = Sequential()
        # Wrap in KeepFirst so that we don't forward skip channels to the toRGB module
        prev_torgb_block.add_module(
            'low_resl_to_rgb',
            TruncateChannels(low_resl_to_rgb, self.get_features_for_level(self.level - 1))
        )

        # Add an upsample layer to match the output resolution of the next block
        prev_torgb_block.add_module(
            'low_resl_upsample',
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=True),
        )

        # Build a new convolutional layer
        inter_block = TruncateChannels(self.gen_conv_block(
            self.get_features_for_level(self.level - 1) + self.get_skip_features_for_level(self.level),
            self.get_features_for_level(self.level)
        ), self.get_features_for_level(self.level - 1) + self.get_skip_features_for_level(self.level))

        # Add a new toRGB block to this conv. layer
        next_torgb_block = Sequential()
        next_torgb_block.add_module('high_resl_block', inter_block)
        next_torgb_block.add_module('high_resl_to_rgb', self.to_rgb_block(self.get_features_for_level(self.level)))

        # Fadein between upsampled output of previous block and output of the new block
        new_model.add_module('output_concat_block', ConcatTable(prev_torgb_block, next_torgb_block))
        new_model.add_module('output_fadein_block', fadein_layer())

        self.model = new_model
        self.module_names = get_module_names(self.model)

    def update_alpha(self, alpha):
        self.model.input_fadein_block.update_alpha(alpha)
        self.model.output_fadein_block.update_alpha(alpha)

    def from_rgb_block(self, c_out):
        layers = conv(self.config.input_channels,
                      c_out,
                      kernel_size=1,
                      stride=1,
                      bias=self.config.need_bias,
                      pad=self.config.pad)
        layers.add_module("from_rgb_batchnorm", norm(c_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            layers.add_module("from_rgb_act", act(self.config.act_fun))
        return layers

    def to_rgb_block(self, c_in):
        layers = conv(c_in,
                      self.config.output_channels,
                      1,
                      bias=self.config.need_bias,
                      pad=self.config.pad)

        if self.config.need_sigmoid:
            layers.add_module('to_rgb_sigmoid', nn.Sigmoid())

        self.input_channels = c_in

        return layers

    def dec_conv_block(self, ndim_in, ndim_out):
        """Returns a convolutional layer for the decoder"""
        conv_block = Sequential()
        conv_block.add(conv(ndim_in,
                            ndim_out,
                            self.config.filter_size_down,
                            2,
                            bias=self.config.need_bias,
                            pad=self.config.pad,
                            downsample_mode=self.config.downsample_mode))
        conv_block.add(norm(ndim_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            conv_block.add(act(self.config.act_fun))
        conv_block.add(conv(ndim_out,
                            ndim_out,
                            self.config.filter_size_down,
                            1,
                            bias=self.config.need_bias,
                            pad=self.config.pad))
        conv_block.add(norm(ndim_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            conv_block.add(act(self.config.act_fun))

        return conv_block

    def gen_conv_block(self, ndim_in, ndim_out):
        """Returns a convolutional layer for the generator"""
        conv_block = Sequential()
        # Upsample layer: doubles resolution, keep same number of channels
        conv_block.add(nn.Upsample(scale_factor=2, mode=self.config.upsample_mode))
        conv_block.add(norm(ndim_in, norm_fun=self.config.norm_fun))

        conv_block.add(conv(ndim_in,
                            ndim_out,
                            kernel_size=self.config.filter_size_up,
                            stride=1,
                            bias=self.config.need_bias,
                            pad=self.config.pad))
        conv_block.add(norm(ndim_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            conv_block.add(act(self.config.act_fun))

        conv_block.add(conv(ndim_out,
                            ndim_out,
                            kernel_size=self.config.filter_size_up,
                            stride=1,
                            bias=self.config.need_bias,
                            pad=self.config.pad))
        conv_block.add(norm(ndim_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            conv_block.add(act(self.config.act_fun))

        if self.config.need1x1_up:
            conv_block.add(conv(ndim_out,
                                ndim_out,
                                1,
                                bias=self.config.need_bias,
                                pad=self.config.pad))
            conv_block.add(norm(ndim_out, norm_fun=self.config.norm_fun))
            if self.config.norm_fun != "InPlaceABN":
                conv_block.add(act(self.config.act_fun))

        return conv_block

    def skip_block(self, ndim_in, ndim_out):
        if not ndim_out:
            return None
        skip = Sequential()
        skip.add(conv(ndim_in, ndim_out, self.config.filter_size_skip, bias=self.config.need_bias, pad=self.config.pad))
        skip.add(norm(ndim_out, norm_fun=self.config.norm_fun))
        if self.config.norm_fun != "InPlaceABN":
            skip.add(act(self.config.act_fun))

        return skip
