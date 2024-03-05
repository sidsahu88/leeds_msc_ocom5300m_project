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

    return ci.squeeze()         # Return ci Shape: (n_channels)


def get_channels_to_preserve(prune_threshold, channels_ci_scores, channels_ccm, caps_dim=8, layers_to_prune = {'conv_layer':0}):
    channels_to_preserve = {}

    for layer, index in layers_to_prune.items():
        layer_threshold = len(channels_ci_scores[index]) * prune_threshold
        layer_threshold -= layer_threshold % caps_dim
        
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
                
        assert len(channels_to_prune) == layer_threshold, "Number of channels: {} selected for pruning for layer: {} is not equal to the threshold: {}".format(len(channels_to_prune), layer, layer_threshold)  
        
        channels_to_preserve[layer] = list(set(np.arange(len(layer_ci_scores))) - set(channels_to_prune))

    return channels_to_preserve
    

def build_pruned_capsnet(orig_model, input_channels, n_class, n_caps_layers, prim_caps_dim, n_preserved_channels):
    import copy

    orig_model_state_dict = orig_model.state_dict()
    new_model = copy.deepcopy(orig_model)

    new_pcaps_layer_n_channels = 0
    curr_conv_input_channels = input_channels

    for layer_name, n_out_channels in n_preserved_channels.items():
        if layer_name.startswith('conv_layer'):
            existing_kernel_size = orig_model_state_dict[layer_name+'.conv_layer.weight'].shape[-1]

            setattr(new_model, layer_name, caps.ConvLayer(in_channels=curr_conv_input_channels,
                                                          out_channels=n_out_channels,
                                                          kernel_size=existing_kernel_size,
                                                          stride=1,
                                                          batch_norm=True))
            
            curr_conv_input_channels = n_out_channels

        elif layer_name.startswith('primary_caps'):
            existing_kernel_size = orig_model_state_dict[layer_name+'.primary_caps.weight'].shape[-1]
            new_pcaps_layer_n_channels = int(n_out_channels/prim_caps_dim)

            setattr(new_model, layer_name, caps.PrimaryCapsLayer(in_channels=curr_conv_input_channels,
                                                                 n_out_caps=new_pcaps_layer_n_channels,
                                                                 out_caps_dim=prim_caps_dim,
                                                                 kernel_size=existing_kernel_size,
                                                                 stride=1))
    if n_caps_layers != 0:
        prev_n_prim_caps = orig_model_state_dict['intermediate_caps.intermediate_caps_layer_1.weights'].shape[0]
    else:
        prev_n_prim_caps = orig_model_state_dict['class_caps.weights'].shape[0]
        
    prev_prim_caps_out_channels = orig_model_state_dict['primary_caps.primary_caps.weight'].shape[0]

    feature_dims = int((prev_n_prim_caps*prim_caps_dim)/ prev_prim_caps_out_channels)

    new_n_prim_caps = feature_dims * new_pcaps_layer_n_channels

    if n_caps_layers != 0:
        intermediate_caps = torch.nn.Sequential()
        intermediate_caps.add_module("intermediate_caps_layer_1",
                                     caps.CapsLayer(caps_layer_name='intermediate_caps_layer_1',
                                                    n_in_caps=new_n_prim_caps,
                                                    n_out_caps=16,
                                                    in_caps_dim=prim_caps_dim,
                                                    out_caps_dim=12))
        if n_caps_layers > 1:
            for caps_layer in range(2, n_caps_layers+1):
                intermediate_caps.add_module("intermediate_caps_layer_" + str(caps_layer),
                                              caps.CapsLayer(caps_layer_name='intermediate_caps_layer_' + str(caps_layer),
                                                             n_in_caps=16,
                                                             n_out_caps=16,
                                                             in_caps_dim=12,
                                                             out_caps_dim=12))

        setattr(new_model, "intermediate_caps", intermediate_caps)

        setattr(new_model, "class_caps", caps.CapsLayer(caps_layer_name='class_caps_layer',
                                                        n_in_caps=16,
                                                        n_out_caps=n_class,
                                                        in_caps_dim=12,
                                                        out_caps_dim=16))

    else:
        setattr(new_model, "class_caps", caps.CapsLayer(caps_layer_name='class_caps_layer',
                                                        n_in_caps=new_n_prim_caps,
                                                        n_out_caps=n_class,
                                                        in_caps_dim=prim_caps_dim,
                                                        out_caps_dim=16))

    return new_model
    

def prune_multilayer_capsnet_layers(orig_model, input_channels, n_class, n_caps_layers, prim_caps_dim, model_layers_to_prune, channels_to_preserve, device='cuda'):
    new_model = build_pruned_capsnet(orig_model, input_channels=input_channels, n_class=n_class, n_caps_layers=n_caps_layers, prim_caps_dim=prim_caps_dim,
                                     n_preserved_channels={layer: len(channels) for layer, channels in channels_to_preserve.items()})
    new_model.to(device)
    print("Pruned Model:", new_model.eval())

    orig_model_state_dict = orig_model.state_dict()
    new_model_state_dict = new_model.state_dict()

    prev_layer_channels = np.arange(input_channels)

    for layer_name in model_layers_to_prune:
        preserve_layer_channels = channels_to_preserve[layer_name]

        if layer_name.startswith('conv_layer'):
            layer_weight_name = layer_name+'.conv_layer.weight'

            new_layer_weight = torch.index_select(orig_model_state_dict[layer_weight_name], 0, torch.tensor(preserve_layer_channels, device=device))
            new_layer_weight = torch.index_select(new_layer_weight, 1, torch.tensor(prev_layer_channels, device=device))
            
            new_model_state_dict[layer_weight_name] = new_layer_weight

            layer_bias_name = layer_name+'.conv_layer.bias'
            if layer_bias_name in orig_model_state_dict.keys():
                new_layer_bias = torch.index_select(orig_model_state_dict[layer_bias_name], 0, torch.tensor(preserve_layer_channels, device=device))
                new_model_state_dict[layer_bias_name] = new_layer_bias
        
            prev_layer_channels = preserve_layer_channels
        
        elif layer_name.startswith('primary_caps'):
            layer_weight_name = layer_name+'.primary_caps.weight'

            new_layer_weight = torch.index_select(orig_model_state_dict[layer_weight_name], 0, torch.tensor(preserve_layer_channels, device=device))
            new_layer_weight = torch.index_select(new_layer_weight, 1, torch.tensor(prev_layer_channels, device=device))

            new_model_state_dict[layer_weight_name] = new_layer_weight

            layer_bias_name = layer_name+'.primary_caps.bias'
            new_layer_bias = torch.index_select(orig_model_state_dict[layer_bias_name], 0, torch.tensor(preserve_layer_channels, device=device))
            new_model_state_dict[layer_bias_name] = new_layer_bias

    new_model.load_state_dict(new_model_state_dict)
    
    return new_model
    

def create_pruned_model_from_state_dict(new_model, saved_state_dict, input_channels, n_class, n_caps_layers, prim_caps_dim):
    n_preserved_channels = {}

    for layer in new_model.state_dict().keys():
        if layer.endswith('conv_layer.weight') or layer.endswith('primary_caps.weight'):
            n_preserved_channels[layer.split('.')[0]] = saved_state_dict[layer].shape[0]

    new_model = build_pruned_capsnet(new_model, input_channels=input_channels, n_class=n_class, n_caps_layers=n_caps_layers,
                                     prim_caps_dim=prim_caps_dim, n_preserved_channels=n_preserved_channels)
    
    new_model_state_dict = new_model.state_dict()

    for layer in new_model_state_dict.keys():
        if not (layer.startswith('intermediate_caps') or layer.startswith('class_caps')):
            new_model_state_dict[layer] = saved_state_dict[layer]

    new_model.load_state_dict(new_model_state_dict)

    return new_model

