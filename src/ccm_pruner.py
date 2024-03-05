import os
import torch
import numpy as np

import capsule_network as caps


def calc_ccm_loss(feature_map):
    feature_map = feature_map.view(feature_map.size(0), feature_map.size(1), -1)    # feature_map shape: (batch_size, channels, feature_map_height * feature_map_width)
    mean_feature_map = feature_map.mean(dim=0)                                      # mean_feature_map shape: (channels, feature_map_height*feature_map_width)
    ccm = torch.corrcoef(mean_feature_map)                                          # ccm shape: (channels, channels)
    ccm_loss = ccm.abs().sum() / ccm.numel()                                        # ccm_loss shape: (1)
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


def get_channels_to_preserve(prune_threshold, channels_ci_scores, prim_caps_dim, layers_to_prune = {'conv_layer':0}):
    channels_to_preserve = {}

    for layer, index in layers_to_prune.items():
        layer_threshold = int(len(channels_ci_scores[index]) * prune_threshold)

        if (layer_threshold > prim_caps_dim):
            layer_threshold -= (layer_threshold) % prim_caps_dim
        else:
            layer_threshold = prim_caps_dim
        
        layer_ci_scores = channels_ci_scores[index]

        ci_sorted_channels = np.argsort(layer_ci_scores)

        channels_to_prune = []

        for channel in ci_sorted_channels:
            if channel not in channels_to_prune:
                channels_to_prune.append(channel)

            if len(channels_to_prune) == layer_threshold:
                break
                
        assert len(channels_to_prune) == layer_threshold, "{} channels selected of the {} layer for pruning does not equal to the threshold: {}".format(len(channels_to_prune), layer, layer_threshold)  
        
        channels_to_preserve[layer] = list(set(np.arange(len(layer_ci_scores))) - set(channels_to_prune))

    return channels_to_preserve
    

def build_pruned_capsnet(orig_model, model_conv_config_dict, n_class, n_caps_layers, prim_caps_dim, preserved_channels_count_dict, squash_fn, n_inter_caps=None):
    import copy

    orig_model_state_dict = orig_model.state_dict()
    new_model = copy.deepcopy(orig_model)

    new_prim_caps_channels = 0
    curr_conv_in_channels = model_conv_config_dict['in_img_c']

    config_indx = 0

    for layer_name, n_out_channels in preserved_channels_count_dict.items():
        if layer_name.startswith('conv_layer'):
            setattr(new_model, layer_name, caps.ConvLayer(in_channels=curr_conv_in_channels,
                                                          out_channels=n_out_channels,
                                                          kernel_size=model_conv_config_dict['kernel_size'][config_indx],
                                                          stride=model_conv_config_dict['stride'][config_indx],
                                                          padding=model_conv_config_dict['padding'][config_indx],
                                                          batch_norm=True))
            
            curr_conv_in_channels = n_out_channels
            
            config_indx += 1

        elif layer_name.startswith('primary_caps'):
            new_prim_caps_channels = int(n_out_channels/prim_caps_dim)

            setattr(new_model, layer_name, caps.PrimaryCapsLayer(in_channels=curr_conv_in_channels,
                                                                 n_out_caps=new_prim_caps_channels,
                                                                 out_caps_dim=prim_caps_dim,
                                                                 kernel_size=model_conv_config_dict['kernel_size'][config_indx],
                                                                 stride=model_conv_config_dict['stride'][config_indx],
                                                                 padding=model_conv_config_dict['padding'][config_indx],
                                                                 squash_fn=squash_fn))
            
            config_indx += 1

    if n_inter_caps is None:
        n_out_intercaps = new_prim_caps_channels
    else:
        n_out_intercaps = n_inter_caps

    if n_caps_layers != 0:
        intermediate_caps = torch.nn.Sequential()
        
        intermediate_caps.add_module("intermediate_convcaps_layer_1",
                                     caps.ConvCapsLayer(n_in_caps = new_prim_caps_channels,
                                                        n_out_caps = n_out_intercaps,
                                                        in_caps_dim = prim_caps_dim,
                                                        out_caps_dim = prim_caps_dim,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        squash_fn=squash_fn))
        
        n_in_intercaps = n_out_intercaps
        
        for conv_caps_layer in range(2, n_caps_layers+1):            
            intermediate_caps.add_module("intermediate_convcaps_layer_" + str(conv_caps_layer),
                                         caps.ConvCapsLayer(n_in_caps = n_in_intercaps,
                                                            n_out_caps = n_out_intercaps,
                                                            in_caps_dim = prim_caps_dim,
                                                            out_caps_dim = prim_caps_dim,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1,
                                                            squash_fn=squash_fn))

        setattr(new_model, "intermediate_caps", intermediate_caps)

    prim_caps_out_dim1, prim_caps_out_dim2 = caps.get_n_conv_dim(len(model_conv_config_dict['kernel_size']), 
                                                                 img_h=model_conv_config_dict['in_img_h'], 
                                                                 img_w=model_conv_config_dict['in_img_w'], 
                                                                 kernel_size=model_conv_config_dict['kernel_size'],
                                                                 stride=model_conv_config_dict['stride'], 
                                                                 padding=model_conv_config_dict['padding'])
    
    new_n_in_classcaps = n_out_intercaps * prim_caps_out_dim1 * prim_caps_out_dim2

    setattr(new_model, "class_caps", caps.CapsLayer(caps_layer_name='class_caps_layer',
                                                    n_in_caps=new_n_in_classcaps,
                                                    n_out_caps=n_class,
                                                    in_caps_dim=prim_caps_dim,
                                                    out_caps_dim=16,
                                                    squash_fn=squash_fn))

    return new_model
    

def prune_capsnet(orig_model, model_conv_config_dict, n_class, n_caps_layers, prim_caps_dim, model_layers_to_prune, preserved_channels_dict, 
                  n_inter_caps=None, squash_fn=caps.squash, device='cuda'):
    
    new_model = build_pruned_capsnet(orig_model, model_conv_config_dict, n_class=n_class, n_caps_layers=n_caps_layers, prim_caps_dim=prim_caps_dim, 
                                     preserved_channels_count_dict={layer: len(channels) for layer, channels in preserved_channels_dict.items()},
                                     squash_fn=squash_fn, n_inter_caps=n_inter_caps)
    new_model.to(device)
    print("Pruned Model:", new_model.eval())

    orig_model_state_dict = orig_model.state_dict()
    new_model_state_dict = new_model.state_dict()

    prev_layer_channels = np.arange(model_conv_config_dict['in_img_c'])

    for layer_name in model_layers_to_prune:
        preserve_layer_channels = preserved_channels_dict[layer_name]

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
    

def create_pruned_model_from_state_dict(new_model, saved_state_dict, model_conv_config_dict, n_class, n_caps_layers, prim_caps_dim, n_inter_caps=None,
                                        base_capsnet_stdict=True, squash_fn=caps.squash):
    n_preserved_channels = {}

    for layer in new_model.state_dict().keys():
        if layer.endswith('conv_layer.weight') or layer.endswith('primary_caps.weight'):
            n_preserved_channels[layer.split('.')[0]] = saved_state_dict[layer].shape[0]

    new_model = build_pruned_capsnet(new_model, model_conv_config_dict, n_class=n_class, n_caps_layers=n_caps_layers, prim_caps_dim=prim_caps_dim,
                                     preserved_channels_count_dict=n_preserved_channels, squash_fn=squash_fn, n_inter_caps=n_inter_caps)

    if base_capsnet_stdict:
        new_model_state_dict = new_model.state_dict()

        for layer in new_model_state_dict.keys():
            if not (layer.startswith('intermediate_caps') or layer.startswith('class_caps')):
                new_model_state_dict[layer] = saved_state_dict[layer]

        new_model.load_state_dict(new_model_state_dict)

    else:
        new_model.load_state_dict(saved_state_dict)

    return new_model

