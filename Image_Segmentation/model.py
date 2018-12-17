import torch.nn as nn


def bn(out_planes):
    return nn.BatchNorm2d(num_features = out_planes)

def relu():
    return nn.ReLU(True)

def conv_bn_relu(in_planes, out_planes, kernel = 3, stride = 2, padding = 1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride, padding = padding),
                         bn(out_planes),
                         relu())

def downsample(in_planes, out_planes, kernel = 3, stride = 2, padding = 1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride, padding = padding),
                         bn(out_planes),
                         relu())

def upsample(in_planes, out_planes, kernel = 3, stride = 2, padding = 1, output_padding = 0):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size = kernel, stride = stride, padding = padding, output_padding = output_padding),
                         bn(out_planes),
                         relu())

def max_pool(out_planes, kernel = 3, stride = 2, padding = 1):
    return nn.MaxPool2d(kernel_size = kernel, stride = stride, padding = padding, return_indices = True)                       

def max_unpool(out_planes, kernel = 3, stride = 2, padding = 1):
    return nn.MaxUnpool2d(kernel_size = kernel, stride = stride, padding = padding)
                     
      
    
# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        self.in_size = in_size
        
        # вход в шаблон: изменяем разрешение HxW на H/2xW/2
        self.down_from_default = downsample(in_planes = self.in_size, out_planes = 32)
        
        self.conv1_1 = conv_bn_relu(in_planes = 32, out_planes = 32, stride = 1)
        self.max_pool1_1 = max_pool(out_planes = 32)
        self.bn1_1 = bn(out_planes = 32)
        self.relu1_1 = relu()
        
        self.conv2_1 = conv_bn_relu(in_planes = 32, out_planes = 64, stride = 1)
        self.conv2_2 = conv_bn_relu(in_planes = 64, out_planes = 64, stride = 1)
        self.max_pool2_1 = max_pool(out_planes = 64, stride = 1)
        self.bn2_1 = bn(out_planes = 64)
        self.relu2_1 = relu()
        
        self.conv3_1 = conv_bn_relu(in_planes = 64, out_planes = 128, stride = 1)
        self.conv3_2 = conv_bn_relu(in_planes = 128, out_planes = 64, stride = 1)
        
        self.max_unpool4_1 = max_unpool(out_planes = 64, stride = 1)
        self.bn4_1 = bn(out_planes = 64)
        self.relu4_1 = relu()
        self.conv4_1 = conv_bn_relu(in_planes = 64, out_planes = 64, stride = 1)
        self.conv4_2 = conv_bn_relu(in_planes = 64, out_planes = 32, stride = 1)
        
        self.max_unpool5_1 = max_unpool(out_planes = 32, kernel = 4)
        self.bn5_1 = bn(out_planes = 32)
        self.relu5_1 = relu()
        self.conv5_1 = conv_bn_relu(in_planes = 32, out_planes = 32, stride = 1)
        
        # выход из шаблона: возвращемся к разрешению HxW перед софтмаксом
        self.up_to_default = upsample(in_planes = 32, out_planes = 2, output_padding = 1)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        
        input = self.down_from_default(input)
        
        input = self.conv1_1(input)
        input, indices_1 = self.max_pool1_1(input)
        input = self.bn1_1(input)
        input = self.relu1_1(input)
        
        input = self.conv2_1(input)
        input = self.conv2_2(input)
        
        input, indices_2 = self.max_pool2_1(input)
        input = self.bn2_1(input)
        input = self.relu2_1(input)
        
        input = self.conv3_1(input)
        input = self.conv3_2(input)

        input = self.max_unpool4_1(input, indices_2)
        input = self.bn4_1(input)
        input = self.relu4_1(input)
        input = self.conv4_1(input)
        input = self.conv4_2(input)

        input = self.max_unpool5_1(input, indices_1)
        input = self.bn5_1(input)
        input = self.relu5_1(input)
        
        input = self.conv5_1(input)
        
        input = self.up_to_default(input)
        
        return self.softmax(input)
        
