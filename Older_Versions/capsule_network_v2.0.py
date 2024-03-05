import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CapsuleNet is using device:', device)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9, stride=1, *args, **kwargs):
        super(ConvLayer, self).__init__(*args, **kwargs)

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return F.relu(self.conv_layer(x))


def get_conv_out_dim(img_h, img_w, kernel_size, stride, padding=0):
    out_h = (int((img_h + (2*padding) - kernel_size) / stride)) + 1
    out_w = (int((img_w + (2*padding) - kernel_size) / stride)) + 1
    return out_h, out_w


def squash(tensor, dim=-1, epsilon=1e-7):
    sqr_norm = torch.sum(tensor**2, dim=dim, keepdim=True)
    norm = torch.sqrt(sqr_norm + epsilon)
    return (sqr_norm * tensor) / ((1. + sqr_norm) * norm)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels=256, out_channels=32, out_caps_dim=8, kernel_size=9, stride=2, padding='valid', *args, **kwargs):
        super(PrimaryCapsLayer, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_caps_dim = out_caps_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_prim_caps = None

        self.primary_caps = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
                for _ in range(out_caps_dim)])

    def forward(self, x):
        if self.n_prim_caps is None:
            conv_out_dim1, conv_out_dim2 = get_conv_out_dim(x.size(-2),             # Calculate convolutional output dimension for zero padding
                                                            x.size(-1),
                                                            self.kernel_size,
                                                            self.stride)
            self.n_prim_caps = conv_out_dim1 * conv_out_dim2 * self.out_channels    # Get number of output capsules

        batch_size = x.size(0)
        caps_output = torch.stack([capsule(x) for capsule in self.primary_caps], dim=-1)      # caps_output shape: (batch_size, out_channels, conv_out_dim1, conv_out_dim2, out_caps_dim)
        u = caps_output.view(batch_size, self.n_prim_caps, -1)                                # Reshape the tensor to (batch_size, n_prim_caps, out_caps_dim)
        return squash(u), caps_output                                                         # Squash returns the tensor shape: (batch_size, n_prim_caps, out_caps_dim)


class ClassCapsLayer(nn.Module):
    def __init__(self, n_class, n_prim_caps, in_caps_dim=8, out_caps_dim=16, *args, **kwargs):
        super(ClassCapsLayer, self).__init__(*args, **kwargs)

        self.n_class = n_class
        self.n_prim_caps = n_prim_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim

        self.weights = nn.Parameter(torch.randn(n_prim_caps, n_class, out_caps_dim, in_caps_dim))

    def forward(self, u):
        batch_size = u.size(0)
        W = torch.stack([self.weights] * batch_size, dim=0)         # W Shape: (batch_size, n_prim_caps, n_class, out_caps_dim, in_caps_dim)
        # print("W shape: ", W.size())
        
        u = torch.stack([u] * self.n_class, dim=2).unsqueeze(-1)    # u Shape: (batch_size, n_prim_caps, n_class, in_caps_dim, 1)
        # print("u shape: ", u.size())
        return torch.matmul(W, u)                                   # return Shape: (batch_size, n_prim_caps, n_class, out_caps_dim, 1)


class RoutingByAggreement(nn.Module):
    def __init__(self, n_in_caps, n_out_caps,  n_iterations=3, *args, **kwargs):
        super(RoutingByAggreement, self).__init__(*args, **kwargs)

        self.n_in_caps = n_in_caps
        self.n_out_caps = n_out_caps
        self.n_iterations = n_iterations

    def forward(self, u_hat):
        # print("u_hat: ", u_hat.size())
        batch_size = u_hat.size(0)                                  # u_hat Shape: (batch_size, n_in_caps, n_out_caps, output_capsule_dim, 1)

        b = torch.zeros(batch_size, self.n_in_caps, self.n_out_caps, 1, 1, 
                        dtype=torch.float, device=device) # b Shape: (batch_size, n_in_caps, n_out_caps, 1, 1)
        # print("b shape: ", b.shape)

        for r in range(self.n_iterations):
            c = F.softmax(b, dim=1)                                 # c Shape: (batch_size, n_in_caps, n_out_caps, 1, 1)
            # print("c shape: ", c.shape)
            
            s = (c * u_hat).sum(dim=1, keepdim=True)                # s Shape: (batch_size, 1, n_out_caps, output_capsule_dim, 1)
            # print("s shape: ", s.shape)
            
            v = squash(s, dim=-2)                                   # v Shape: (batch_size, 1, n_out_caps, output_capsule_dim, 1)
            
            if r < self.n_iterations - 1:
                a = torch.matmul(u_hat.transpose(-2, -1), torch.cat([v] * self.n_in_caps, dim=1))    # a Shape: (batch_size, n_in_caps, n_out_caps, 1, 1)
                # print("a shape: ", a.shape)
                b = b + a

        # print("v squeezed shape: ", v.squeeze(1).shape)
        return v.squeeze(1)         # v Shape Squeezed: (batch_size, n_out_caps, output_capsule_dim, 1)


def get_v_norm(v):
    return torch.sqrt((v**2).sum(dim=-2)).squeeze()     # v_norm Shape: (batch_size, n_class)


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
        # print("v shape: ", v.shape)
        caps_out_class = get_v_norm(v)                      # caps_out_class shape: (batch_size, n_class)
        # print("caps_out_class shape:", caps_out_class.shape)
        # print("caps_out_class: ", caps_out_class[0])
        caps_out_class = F.softmax(caps_out_class, dim=1)   # caps_out_class Shape: (batch_size, n_class)
        # print("caps_out_class shape after softmax:", caps_out_class.shape)
        # print("caps_out_class: ", caps_out_class[0])
        
        masked_matrix = torch.eye(self.n_class, device=device)
        masked_matrix = masked_matrix.index_select(dim=0, index=caps_out_class.argmax(dim=1).squeeze())       # masked_matrix Shape: (batch_size, n_class)

        v = v * masked_matrix[:, :, None, None]         # v Shape: (batch_size, n_class, out_capsule_dim, 1)
        v = v.view(v.size(0), -1)                       # flattened v Shape: (batch_size, n_class * out_capsule_dim * 1)
        reconstructed_img = self.fully_conn_layers(v)   # reconstructed_img Shape: (batch_size, fcl_ouput_dim)
        
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
        batch_size = v.size(0)                                      # v Shape: (batch_size, n_out_caps, output_capsule_dim, 1)
        
        present_error = F.relu(self.m_positive - get_v_norm(v))**2  # present_error Shape: (batch_size, n_class)
        absent_error = F.relu(get_v_norm(v) - self.m_negative)**2   # absent_error Shape: (batch_size, n_class)
        # print("present_error shape:", present_error.shape)
        # print("absent_error shape:", absent_error.shape)
        
        labels = F.one_hot(labels, self.n_class)

        loss = (labels * present_error) + (self.lmbda * (1 - labels) * absent_error)
        loss = loss.sum(dim=1).mean()
        return loss

    def reconstruction_loss(self, images, reconstructed_images):
        return self.mse_loss(reconstructed_images, images) * self.recon_loss_scale_factor


class BaseCapsuleNetwork(nn.Module):
    def __init__(self, in_img_c, in_img_h, in_img_w, n_class, *args, **kwargs):
        super(BaseCapsuleNetwork, self).__init__(*args, **kwargs)
        
        self.name = 'BaseCapsuleNetwork'

        self.conv_layer = ConvLayer(in_channels=in_img_c, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsLayer(out_channels=32, kernel_size=9, stride=2)

        conv_layer_out_dim1, conv_layer_out_dim2 = get_conv_out_dim(in_img_h, in_img_w, kernel_size=9, stride=1)
        prim_caps_out_dim1, prim_caps_out_dim2 = get_conv_out_dim(conv_layer_out_dim1, conv_layer_out_dim2, kernel_size=9, stride=2)
        n_prim_caps = 32 * prim_caps_out_dim1 * prim_caps_out_dim2

        self.class_caps = ClassCapsLayer(n_class=n_class, n_prim_caps=n_prim_caps)
        self.routing_aggreement = RoutingByAggreement(n_in_caps=n_prim_caps, n_out_caps=n_class)
        self.caps_decoder = CapsDecoder(n_class, in_img_c, in_img_h, in_img_w)

    def forward(self, images):
        layer_feature_maps = []
        
        output = self.conv_layer(images)
        layer_feature_maps.append(output.detach().clone())
        
        u, caps_output = self.primary_caps(output)
        layer_feature_maps.append(caps_output.detach().clone())
        
        u = self.class_caps(u)
        pred = self.routing_aggreement(u)
        reconstructed_img = self.caps_decoder(pred)
        
        return reconstructed_img, pred, layer_feature_maps

