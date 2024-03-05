import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CapsuleNet is using device:', device)


def get_conv_out_dim(img_h, img_w, kernel_size, stride=1, padding=0):
    
    out_h = (int((img_h + (2*padding) - kernel_size) / stride)) + 1
    out_w = (int((img_w + (2*padding) - kernel_size) / stride)) + 1
    
    return out_h, out_w


def get_n_conv_dim(n_conv_layers, img_h, img_w, kernel_size=[0], stride=[1], padding=[0]):
    
    assert n_conv_layers == len(kernel_size) and n_conv_layers == len(stride) and n_conv_layers == len(padding), "Number of convolutional layers does not matches the size of kernel/stride/padding."

    out_h = img_h
    out_w = img_w

    for layer in range(n_conv_layers):
        out_h, out_w = get_conv_out_dim(out_h, out_w, kernel_size[layer], stride[layer], padding[layer])
        
    return out_h, out_w


def squash(tensor, dim=-1, epsilon=1e-06):

    sqr_norm = torch.sum(tensor**2, dim=dim, keepdim=True)
    norm = torch.sqrt(sqr_norm+epsilon)
    
    return (sqr_norm * tensor) / ((1. + sqr_norm) * norm)
    

def sensitive_squash(tensor, dim=-1, epsilon=1e-06):
    
    sqr_norm = torch.sum(tensor**2, dim=dim, keepdim=True)
    norm = torch.sqrt(sqr_norm+epsilon)
    
    return (1 - 1/torch.exp(norm)) * (tensor/norm)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9, stride=1, padding='valid', batch_norm=False, *args, **kwargs):
        
        super(ConvLayer, self).__init__(*args, **kwargs)
        
        self.batch_norm = batch_norm
           
        if batch_norm:
            self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, bias=False)
            
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding)

    def forward(self, x):
        
        x = self.conv_layer(x)
        
        if self.batch_norm:
            x = self.bn(x)
        
        return F.relu(x)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels=256, n_out_caps=32, out_caps_dim=8, kernel_size=9, stride=1, padding='valid', squash_fn=squash, *args, **kwargs):

        super(PrimaryCapsLayer, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.n_out_caps = n_out_caps
        self.out_caps_dim = out_caps_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.squash_fn = squash_fn
        self.n_prim_caps = None

        self.primary_caps = nn.Conv2d(in_channels=in_channels, out_channels=n_out_caps*out_caps_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):

        if self.n_prim_caps is None:
            conv_out_dim1, conv_out_dim2 = get_conv_out_dim(x.size(-2),         # Calculate convolutional output dimension for zero padding
                                                            x.size(-1),
                                                            self.kernel_size,
                                                            self.stride)
            self.n_prim_caps = conv_out_dim1 * conv_out_dim2 * self.n_out_caps  # Get number of output capsules

        caps_output = self.primary_caps(x)                                      # caps_output shape: (batch_size, n_out_caps*out_caps_dim, conv_out_dim1, conv_out_dim2)     
        u = caps_output.view(x.size(0), self.n_prim_caps, -1)                   # Reshape the tensor to (batch_size, n_prim_caps, out_caps_dim)
        
        return self.squash_fn(u), caps_output                                   # Squash returns the tensor shape: (batch_size, n_prim_caps, out_caps_dim)


class RoutingByAggreement(nn.Module):
    def __init__(self, n_in_caps, n_out_caps, n_iterations=3, squash_fn=squash, *args, **kwargs):

        super(RoutingByAggreement, self).__init__(*args, **kwargs)

        self.n_in_caps = n_in_caps
        self.n_out_caps = n_out_caps
        self.n_iterations = n_iterations
        self.squash_fn = squash_fn

    def forward(self, u_hat):

        batch_size = u_hat.size(0)                                  # u_hat Shape: (batch_size, n_in_caps, n_out_caps, out_caps_dim)

        b = torch.zeros(batch_size, self.n_in_caps, self.n_out_caps, 1, 
                        dtype=torch.float, device=device)           

        for r in range(self.n_iterations):
            c = F.softmax(b, dim=2)                                 # c Shape: (batch_size, n_in_caps, n_out_caps, 1)
            s = (c * u_hat).sum(dim=1, keepdim=True)                # s Shape: (batch_size, 1, n_out_caps, out_caps_dim)
            
            v = self.squash_fn(s)                                   # v Shape: (batch_size, 1, n_out_caps, out_caps_dim)
            
            if r < self.n_iterations - 1:
                b = b + (v * u_hat).sum(dim=-1, keepdim=True)       # u_hat*v Shape: (batch_size, n_in_caps, n_out_caps, 1)

        return v.view(batch_size, self.n_out_caps, -1)              # v Shape: (batch_size, n_out_caps, out_caps_dim)


class ConvCapsLayer(nn.Module):
    def __init__(self, n_in_caps, n_out_caps, in_caps_dim, out_caps_dim, kernel_size=3, stride=1, padding=0, n_iterations=3, squash_fn=squash, *args, **kwargs):
        
        super(ConvCapsLayer, self).__init__(*args, **kwargs)
        
        self.n_in_caps = n_in_caps
        self.n_out_caps = n_out_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim

        self.kernel_size = kernel_size
        self.in_caps_kernel_size = (kernel_size ** 2) * n_in_caps 

        self.stride = stride
        self.padding = padding

        self.n_iterations = n_iterations
        self.squash_fn = squash_fn
        
        self.weights = nn.Parameter(torch.FloatTensor(self.in_caps_kernel_size, n_out_caps*out_caps_dim, in_caps_dim))
        nn.init.normal_(self.weights, 0, std=0.5)

    def forward(self, u):
        
        batch_size = u.size(0)                                          # u Shape: (batch_size, total_feature_size*n_in_caps, in_caps_dim)
        
        u = u.view(batch_size, self.n_in_caps, self.in_caps_dim, -1)
        
        total_feature_size = u.size(-1)
        feature_size = int(total_feature_size**(1/2))
        
        u = u.view(batch_size, self.n_in_caps*self.in_caps_dim, feature_size, feature_size)
        u = F.unfold(u, self.kernel_size, stride=self.stride, padding=self.padding)
        u = u.view(batch_size, total_feature_size, self.in_caps_kernel_size, self.in_caps_dim, 1)

        u_hat = torch.matmul(self.weights, u).squeeze(-1)
        
        return self.convcaps_routing(u_hat, batch_size, total_feature_size)

    def convcaps_routing(self, u_hat, batch_size, total_feature_size):        
        
        u_hat = u_hat.view(batch_size, total_feature_size, self.in_caps_kernel_size, self.n_out_caps, self.out_caps_dim)

        b = torch.zeros(batch_size, total_feature_size, self.in_caps_kernel_size, self.n_out_caps, 1, 
                        dtype=torch.float, device=device)
        
        for r in range(self.n_iterations):
            c = F.softmax(b, dim=3)
            s = (c * u_hat).sum(dim=2, keepdim=True)                # s shape: (batch_size, total_feature_size, 1, n_out_caps, out_caps_dim)
            
            v = self.squash_fn(s)                                   # v Shape: (batch_size, total_feature_size, 1, n_out_caps, out_caps_dim)
            
            if r < self.n_iterations - 1:
                b = b + (v * u_hat).sum(dim=-1, keepdim=True)
        
        return v.view(batch_size, -1, self.out_caps_dim)            # v Shape: (batch_size, total_feature_size*n_out_caps, out_caps_dim)


class CapsLayer(nn.Module):
    def __init__(self, caps_layer_name, n_in_caps, n_out_caps, in_caps_dim=8, out_caps_dim=16, n_iterations=3, squash_fn=squash, *args, **kwargs):

        super(CapsLayer, self).__init__(*args, **kwargs)
        
        self.caps_layer_name = caps_layer_name
        self.n_out_caps = n_out_caps
        self.n_in_caps = n_in_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim
        self.n_iterations = n_iterations
        self.squash_fn = squash_fn

        self.weights = nn.Parameter(torch.FloatTensor(n_in_caps, n_out_caps, out_caps_dim, in_caps_dim))
        # self.routing = RoutingByAggreement(n_in_caps=n_in_caps, n_out_caps=n_out_caps, squash_fn=squash_fn)
        
        nn.init.normal_(self.weights, 0, std=0.1)
        
    def forward(self, u):

        u = u.view(u.size(0), self.n_in_caps, 1, self.in_caps_dim, 1)   # u Shape: (batch_size, n_in_caps, 1, in_caps_dim, 1)
        u_hat = torch.matmul(self.weights, u)                           # u_hat Shape: (batch_size, n_in_caps, n_out_caps, out_caps_dim, 1)
        
        return self.routing(u_hat.squeeze())                            # Routing return Shape: (batch_size, n_out_caps, out_caps_dim)
        
    def routing(self, u_hat):
        batch_size = u_hat.size(0)                                  # u_hat Shape: (batch_size, n_in_caps, n_out_caps, out_caps_dim)

        b = torch.zeros(batch_size, self.n_in_caps, self.n_out_caps, 1, 
                        dtype=torch.float, device=device)           

        for r in range(self.n_iterations):
            c = F.softmax(b, dim=2)                                 # c Shape: (batch_size, n_in_caps, n_out_caps, 1)
            s = (c * u_hat).sum(dim=1, keepdim=True)                # s Shape: (batch_size, 1, n_out_caps, out_caps_dim)
            
            v = self.squash_fn(s)                                   # v Shape: (batch_size, 1, n_out_caps, out_caps_dim)
            
            if r < self.n_iterations - 1:
                b = b + (v * u_hat).sum(dim=-1, keepdim=True)       # u_hat*v Shape: (batch_size, n_in_caps, n_out_caps, 1)

        return v.view(batch_size, self.n_out_caps, -1)              # v Shape: (batch_size, n_out_caps, out_caps_dim)


class CapsDecoder(nn.Module):
    def __init__(self, n_class, in_img_c, in_img_h, in_img_w, out_capsule_dim=16, *args, **kwargs):
        
        super(CapsDecoder, self).__init__(*args, **kwargs)

        self.n_class = n_class
        self.in_img_c = in_img_c
        self.in_img_h = in_img_h
        self.in_img_w = in_img_w

        fcl_input_dim = out_capsule_dim * n_class
        fcl_ouput_dim = in_img_c * in_img_h * in_img_w

        self.fully_conn_layers = nn.Sequential(
            nn.Linear(fcl_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, fcl_ouput_dim),
            nn.Sigmoid())

    def forward(self, v):
        
        caps_out_class = v.norm(dim=-1)                     # caps_out_class shape: (batch_size, n_class)
        caps_out_class = F.softmax(caps_out_class, dim=1)   # caps_out_class Shape: (batch_size, n_class)
        
        masked_matrix = torch.eye(self.n_class, device=device)
        masked_matrix = masked_matrix.index_select(dim=0, index=caps_out_class.argmax(dim=1).squeeze())       # masked_matrix Shape: (batch_size, n_class)

        v = v * masked_matrix[:, :, None]                   # v Shape: (batch_size, n_class, out_capsule_dim, 1)
        v = v.view(v.size(0), -1)                           # flattened v Shape: (batch_size, n_class * out_capsule_dim * 1)
        reconstructed_img = self.fully_conn_layers(v)       # reconstructed_img Shape: (batch_size, fcl_ouput_dim)
        
        return reconstructed_img.view(-1, self.in_img_c, self.in_img_h, self.in_img_w)   # reconstructed_img Shape: (batch_size, in_img_ch, in_img_h, in_img_w)


class CapsNetLoss(nn.Module):
    def __init__(self, n_class, lmbda=0.5, m_positive=0.9, m_negative=0.1, recon_loss_scale_factor=0.0005, *args, **kwargs):
        
        super(CapsNetLoss, self).__init__(*args, **kwargs)

        self.n_class = n_class
        self.lmbda = lmbda
        self.m_positive = m_positive
        self.m_negative = m_negative
        self.recon_loss_scale_factor = recon_loss_scale_factor
        self.mse_loss = nn.MSELoss()

    def forward(self, v, labels, images=None, reconstructed_images=None):
        
        if images is not None and reconstructed_images is not None:
            return self.margin_loss(v, labels) + self.reconstruction_loss(images, reconstructed_images)
        
        return self.margin_loss(v, labels)

    def margin_loss(self, v, labels):
    
        v = v.norm(dim=-1)                              # v Shape: (batch_size, n_out_caps, output_capsule_dim)
        
        present_error = F.relu(self.m_positive - v)**2  # present_error Shape: (batch_size, n_class)
        absent_error = F.relu(v - self.m_negative)**2   # absent_error Shape: (batch_size, n_class)
        
        labels = F.one_hot(labels, v.size(1))

        loss = (labels * present_error) + (self.lmbda * (1 - labels) * absent_error)
        loss = loss.sum(dim=1).mean()
        
        return loss

    def reconstruction_loss(self, images, reconstructed_images):
        return self.mse_loss(reconstructed_images, images) * self.recon_loss_scale_factor


class CapsuleNetwork(torch.nn.Module):
    def __init__(self, in_img_c, in_img_h, in_img_w, n_class, model_conv_config=None, prim_caps_dim=12, prim_caps_channels=16, n_caps_layers=0, squash_fn=squash, extract_feature_maps=False, *args, **kwargs):
        
        super(CapsuleNetwork, self).__init__(*args, **kwargs)

        self.name = 'CapsuleNetwork'
        
        self.n_caps_layers = n_caps_layers
        self.prim_caps_dim = prim_caps_dim
        self.prim_caps_channels = prim_caps_channels
        self.extract_feature_maps = extract_feature_maps
        
        if model_conv_config is None:
            model_conv_config = {'kernel_size': [3, 3, 3, 3],
                                 'stride': [1, 2, 1, 2],
                                 'padding': [1, 1, 1, 0]}
        
        conv_layer_indx = 0
        
        prim_caps_in_channels = prim_caps_dim * prim_caps_channels

        # Conv Layers
        self.conv_layer_1 = ConvLayer(in_channels=in_img_c, out_channels=int(prim_caps_in_channels/4),
                                      kernel_size=model_conv_config['kernel_size'][conv_layer_indx], stride=model_conv_config['stride'][conv_layer_indx], 
                                      padding=model_conv_config['padding'][conv_layer_indx], batch_norm=True)
        conv_layer_indx += 1
        
        self.conv_layer_2 = ConvLayer(in_channels=int(prim_caps_in_channels/4), out_channels=int(prim_caps_in_channels/2),
                                      kernel_size=model_conv_config['kernel_size'][conv_layer_indx], stride=model_conv_config['stride'][conv_layer_indx], 
                                      padding=model_conv_config['padding'][conv_layer_indx], batch_norm=True)
        conv_layer_indx += 1
        
        self.conv_layer_3 = ConvLayer(in_channels=int(prim_caps_in_channels/2), out_channels=prim_caps_in_channels,
                                      kernel_size=model_conv_config['kernel_size'][conv_layer_indx], stride=model_conv_config['stride'][conv_layer_indx], 
                                      padding=model_conv_config['padding'][conv_layer_indx], batch_norm=True)
        conv_layer_indx += 1
        
        # Primary Capsules
        self.primary_caps = PrimaryCapsLayer(in_channels=prim_caps_in_channels, n_out_caps=prim_caps_channels, out_caps_dim=prim_caps_dim, 
                                             kernel_size=model_conv_config['kernel_size'][conv_layer_indx], stride=model_conv_config['stride'][conv_layer_indx], 
                                             padding=model_conv_config['padding'][conv_layer_indx], squash_fn=squash_fn)

        # Intermediate Capsules
        if n_caps_layers != 0:
            self.intermediate_caps = torch.nn.Sequential()
            for caps_layer in range(1, self.n_caps_layers+1):
                self.intermediate_caps.add_module("intermediate_convcaps_layer_" + str(caps_layer),
                                                  ConvCapsLayer(n_in_caps = prim_caps_channels,
                                                                 n_out_caps = prim_caps_channels,
                                                                 in_caps_dim = prim_caps_dim,
                                                                 out_caps_dim = prim_caps_dim,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 squash_fn=squash_fn))

        prim_caps_out_dim1, prim_caps_out_dim2 = get_n_conv_dim(4, in_img_h, in_img_w, kernel_size=model_conv_config['kernel_size'],
                                                                stride=model_conv_config['stride'], padding=model_conv_config['padding'])
        
        total_caps = prim_caps_channels * prim_caps_out_dim1 * prim_caps_out_dim2

        # Class Capsules
        self.class_caps = CapsLayer(caps_layer_name='class_caps_layer',
                                    n_in_caps=total_caps,
                                    n_out_caps=n_class,
                                    in_caps_dim=prim_caps_dim,
                                    out_caps_dim=16,
                                    squash_fn=squash_fn)

    def forward(self, images):

        layer_feature_maps = []

        if self.extract_feature_maps:
            conv_output = self.conv_layer_1(images)
            layer_feature_maps.append(conv_output.detach().clone())

            conv_output = self.conv_layer_2(conv_output)
            layer_feature_maps.append(conv_output.detach().clone())

            conv_output = self.conv_layer_3(conv_output)
            layer_feature_maps.append(conv_output.detach().clone())

            u, primary_caps_output = self.primary_caps(conv_output)
            layer_feature_maps.append(primary_caps_output.detach().clone())

        else: 
            conv_output = self.conv_layer_1(images)
            conv_output = self.conv_layer_2(conv_output)
            conv_output = self.conv_layer_3(conv_output)
            u, _ = self.primary_caps(conv_output)

        if self.n_caps_layers >= 1:
            u = self.intermediate_caps(u)

        pred = self.class_caps(u)

        return None, pred, layer_feature_maps
