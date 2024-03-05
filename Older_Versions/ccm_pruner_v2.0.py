import os
import torch
import numpy as np

import capsule_network as caps


def calc_ccm_loss(feature_maps):
    mean_feature_maps = feature_maps.mean(dim=0)        # mean_feature_map shape: (channels, feature_map_height, feature_map_width)
    flattened_mean_feat_maps = mean_feature_maps.view(mean_feature_maps.size(0), -1)    # flattened_mean_feat_map shape: (channels, feature_map_height * feature_map_width)
    ccm = torch.corrcoef(flattened_mean_feat_maps)      # ccm shape: (channels, channels)
    ccm = ccm.abs()
    ccm_loss = ccm.sum() / ccm.numel()                  # ccm_loss shape: (1)
    return ccm_loss, ccm


def save_ccm(ccm, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_file = os.path.join(file_path, "{}.pth".format(file_name))
    torch.save(ccm, save_file)
    print("{} saved".format(file_name))


def load_ccm(file_path, file_name):
    save_file = os.path.join(file_path, "{}.pth".format(file_name))
    if not save_file:
        print("Can not find file!")
        return
    ccm = torch.load(save_file)
    return ccm


def reduced_1_row_norm(input, channel):
    input[channel, :] = torch.zeros(input.shape[-1])
    m = torch.linalg.norm(input, ord='nuc', dtype=torch.float32).item()
    return m


def ci_score(feature_maps):
    mean_feature_map = feature_maps.mean(dim=0)
    
    n_channels = mean_feature_map.shape[0]
    
    conv_output = torch.round(mean_feature_map, decimals=4)
    conv_output = conv_output.view(n_channels, -1)

    ci = torch.zeros(1, n_channels)

    r1_norm = torch.zeros(1, n_channels)
    
    for channel in range(n_channels): 
        r1_norm[:, channel] = reduced_1_row_norm(conv_output.detach().clone(), channel)

    original_norm = torch.linalg.norm(conv_output, ord='nuc', dtype=torch.float32).item()
    ci = original_norm - r1_norm

    return ci.squeeze()


def get_channels_to_preserve(threshold, channels_ci_scores, channels_ccm, caps_dim=8, layers_to_prune = {'conv_layer':0}):
    conv_layer_threshold = len(channels_ci_scores[0]) * threshold
    conv_layer_threshold -= conv_layer_threshold % caps_dim

    channels_to_preserve = {}

    for layer, index in layers_to_prune.items():
        if layer.startswith('conv_layer'):
            layer_threshold = conv_layer_threshold
        else:
            layer_threshold = int(conv_layer_threshold / caps_dim)

        layer_ci_scores = channels_ci_scores[index]
        layer_channels_ccm = channels_ccm[index]

        ci_sorted_channels = np.argsort(layer_ci_scores)

        channels_to_prune = []

        for channel in ci_sorted_channels:
            if channel not in channels_to_prune:
                layer_channels_ccm_argsort = torch.argsort(layer_channels_ccm[channel], descending=True)
                most_correlated_channel = layer_channels_ccm_argsort[1].item()

                if most_correlated_channel not in channels_to_prune:
                    channels_to_prune.append(channel)

            if len(channels_to_prune) == layer_threshold:
                break
                
        assert len(channels_to_prune) == layer_threshold, "Number of channels selected: {} for pruning is lesser than the threshold: {}".format(len(channels_to_prune), layer_threshold)  
        
        channels_to_preserve[layer] = list(set(np.arange(len(layer_ci_scores))) - set(channels_to_prune))

    return channels_to_preserve
    

def prune_capsnet_layers(model, input_channels, n_class, layers_to_prune, channels_to_preserve, device='cuda'):
    import copy

    model_state_dict = model.state_dict()
    new_model = copy.deepcopy(model)

    new_conv_layer_n_channels = 0
    new_pcaps_layer_n_channels = 0

    for indx, layer in enumerate(layers_to_prune):
        if layer.startswith('conv_layer'):
            layer_weight_name = layer+'.weight'
            layer_bias_name = layer+'.bias'
            preserve_channels = channels_to_preserve['conv_layer']

            new_layer_weight = torch.index_select(model_state_dict[layer_weight_name], 0, torch.tensor(preserve_channels, device=device))
            new_layer_bias = torch.index_select(model_state_dict[layer_bias_name], 0, torch.tensor(preserve_channels, device=device))

            new_conv_layer_n_channels = len(preserve_channels)
            new_model.conv_layer = caps.ConvLayer(input_channels, new_conv_layer_n_channels)

            new_model.state_dict()[layer_weight_name] = new_layer_weight
            new_model.state_dict()[layer_bias_name] = new_layer_bias

        elif layer.startswith('primary_caps'):
            layer_weight_name = layer+'.weight'
            layer_bias_name = layer+'.bias'
            preserve_channels = channels_to_preserve['primary_caps']

            new_layer_weight = torch.index_select(model_state_dict[layer_weight_name], 0, torch.tensor(preserve_channels, device=device))
            new_layer_bias = torch.index_select(model_state_dict[layer_bias_name], 0, torch.tensor(preserve_channels, device=device))

            new_pcaps_layer_n_channels = len(preserve_channels)
            new_model.primary_caps = caps.PrimaryCapsLayer(new_conv_layer_n_channels, new_pcaps_layer_n_channels)

            new_model.state_dict()[layer_weight_name] = new_layer_weight
            new_model.state_dict()[layer_bias_name] = new_layer_bias

    prev_n_prim_caps = model_state_dict['class_caps.weights'].shape[0]
    prev_prim_caps_out_channels = model_state_dict['primary_caps.primary_caps.weight'].shape[0]

    new_n_prim_caps = int((prev_n_prim_caps / prev_prim_caps_out_channels) * new_pcaps_layer_n_channels)

    new_model.class_caps = caps.ClassCapsLayer(n_class=n_class, n_in_caps=new_n_prim_caps)
    new_model.routing_aggreement = caps.RoutingByAggreement(n_in_caps=new_n_prim_caps, n_out_caps=n_class)

    return new_model
