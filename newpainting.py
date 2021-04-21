from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = 512 if torch.cuda.is_available() else 128

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()])
    # Resize大小
    # 转换为Tensor


def image_loader(image):
    # 添加0维度batch 网络输入
    image = transform(image).unsqueeze(0)
    # 返回Tensor
    return image.to(device, torch.float)

# 使两张图片大小一样（风格迁移算法必须大小一样）
def change_size(style_img, content_img):
    sty_x = style_img.size[0]
    sty_y = style_img.size[1]

    cont_x = content_img.size[0]
    cont_y = content_img.size[1]

    # print(style_img.size)
    # print(content_img.size)
    if sty_x > cont_x:

        sty_x = cont_x
        style_img = style_img.resize((sty_x, sty_y))
        if sty_y > cont_y:
            sty_y = cont_y
            style_img = style_img.resize((sty_x, sty_y))
        else:
            cont_y = sty_y
            content_img = content_img.resize((cont_x, cont_y))

    elif sty_x < cont_x:

        cont_x = sty_x
        content_img = content_img.resize((cont_x, cont_y))

        if sty_y > cont_y:
            sty_y = cont_y
            style_img = style_img.resize((sty_x, sty_y))
        else:
            cont_y = sty_y
            content_img = content_img.resize((cont_x, cont_y))

    return style_img, content_img

def input_image(style_img_path,content_img_path):
    style_img = Image.open(style_img_path)
    content_img = Image.open(content_img_path)
    return style_img,content_img




unloader = transforms.ToPILImage()  # 用于转为PIL

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()

    # 取消Tensor中的维度0
    image = image.squeeze(0)

    # 将Tensor转换为PIL
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)



class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()

        self.target = target.detach()
        # 无梯度Tensor

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input




def gram_matrix(input):
    batch_size, C, H, W = input.size()  # [batch size,channels,H,W]


    features = input.view(batch_size * C, H * W)  # resise

    G = torch.mm(features, features.t())  # 矩阵features * 转置

    return G.div(batch_size * C * H * W)



class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()


mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        # mean std 变为[C x 1 x 1]
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalization
        return (img - self.mean) / self.std



content_layers_default = ['conv_4','conv_5','conv_6','conv_7'] # 第四个conv后计算loss
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] # 每一个conv后计算loss

def get_style_model_and_losses(cnn, mean, std,
                               style_img_Tensor, content_img_Tensor,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)


    normalization = Normalization(mean, std).to(device)


    content_losses = []
    style_losses = []

    # 添加一层normalization
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            #print(name)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # ReLU 的inplace改为False
            layer = nn.ReLU(inplace=False)
            #print(name)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            #print(name)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
            #print(name)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 添加 content loss 至 model 中:
            target = model(content_img_Tensor).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            #print(model)
            content_losses.append(content_loss)
            #print("content",name)

        if name in style_layers:
            # 添加 add style loss 至 model 中:
            target_feature = model(style_img_Tensor).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            #print("style", name)
            #print(model)


    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses







def get_input_optimizer(input_img):

    # LBFGS算法
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer











        #startepoch = (checkpoint['epoch']) + 1


def run_style_transfer(cnn, mean, std,
                       content_img_Tensor, style_img_Tensor, input_img, num_steps=150,
                       style_weight=1e6, content_weight=1):


    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        mean, std, style_img_Tensor, content_img_Tensor)
    optimizer = get_input_optimizer(input_img)


    run = [0]
    #path = './state/Epoch{}model.pth'.format(30)
    #checkpoint = torch.load (path)
    #optimizer.load_state_dict (checkpoint ['optimizer'])
    #model.load_state_dict(checkpoint['model'])
    #startepoch = (checkpoint['epoch']) + 1
    #print(startepoch)
    #while run[0] <= num_steps:
    #startepoch = 0

    for run[0] in range(0,num_steps):

        #checkpoint = torch.load (path)
        #optimizer.load_state_dict (checkpoint ['optimizer'])
        #model.load_state_dict (checkpoint ['model'])
        def clo():
            input_img.data.clamp_(0, 1)
            # 范围在(0,1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            if True:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))


            state = {'model': model.state_dict (), 'optimizer': optimizer.state_dict (), 'epoch': run[0]}
            filepath = './state/' + 'Epoch{:d}'.format (run[0]) + 'model.pth'
            #print (filepath)
            #torch.save (state, filepath)
            #print ('Model of Epoch {:d} has been saved'.format (run[0]))
            return style_score + content_score

        optimizer.step(clo)


    input_img.data.clamp_(0, 1)

    return input_img







if __name__ == '__main__':

    style_img_path = './picture/pic11.jpg'
    content_img_path = './picture/Van4.jpg'

    style_img,content_img = input_image(style_img_path,content_img_path)
    style_img, content_img = change_size(style_img,content_img)

    style_img_Tensor = image_loader(style_img)
    content_img_Tensor = image_loader(content_img)

    assert style_img_Tensor.size() == content_img_Tensor.size(), \
        "风格图与内容图size不一样"

    plt.ion()

    #plt.figure()
    #imshow(style_img_Tensor, title='Style Image')

    #plt.figure()
    #imshow(content_img_Tensor, title='Content Image')

    input_img = content_img_Tensor.clone()

    output = run_style_transfer(cnn, mean, std,
                                content_img_Tensor, style_img_Tensor, input_img)

    plt.figure()
    imshow(output, title='Output Image')


    plt.ioff()
    plt.show()