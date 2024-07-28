
import cv2
import math
from scipy.interpolate import splprep, splev
from chen_tool.bezier import bezier_1 as bezier
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw
from torchvision.transforms.functional import InterpolationMode
import itertools
from PIL import ImageOps
from math import sqrt
class generator():
    def __init__(self, gpu_ids,ngf,init_gain,load_path):
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        self.load_size = 512
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = ngf
        self.init_gain = init_gain
        self.load_path = load_path
        self.real_A = None
        self.fake_B = None
        # torch.backends.cudnn.benchmark = True # 加速卷积，在图像一样大的时候可以加速

        # ############### 初始化生成器 G
        # 定义网络
        net = UnetGenerator(self.input_nc, self.output_nc, 8, self.ngf, use_dropout=False)

        # GPU处理
        if gpu_ids and torch.cuda.is_available():
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # Wrap the network into DataParallel if multiple GPUs are used

        # 初始化权重
        for m in net.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
                init.normal_(m.weight.data, 0.0, self.init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                init.normal_(m.weight.data, 1.0, self.init_gain)
                init.constant_(m.bias.data, 0.0)

        self.netG = net

        # 3 加载pth
        state_dict = torch.load(self.load_path, map_location=str(self.device))

        # ######## 超长防呆设计
        if hasattr(state_dict, '_metadata'): del state_dict._metadata
        for key in list(state_dict.keys()): # 对于使用InstanceNorm的网络层，处理特定的状态键
            parts = key.split('.')
            obj = self.netG
            for part in parts[:-1]:  # 遍历到倒数第二个元素，定位到具体的层
                obj = getattr(obj, part)
            # 检查层的类型，是否是InstanceNorm
            if obj.__class__.__name__.startswith('InstanceNorm'):
                attr = parts[-1]
                # 对于InstanceNorm不应该跟踪的属性，如运行均值和方差，进行特殊处理
                if attr in ['running_mean', 'running_var', 'num_batches_tracked'] and getattr(obj, attr,None) is None:
                    state_dict.pop(key)  # 从状态字典中移除这些键

        # 加载处理后的状态字典到模型中
        self.netG.load_state_dict(state_dict)


class UnetGenerator(nn.Module):
    # 构造函数，初始化生成器
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=False):
        super(UnetGenerator, self).__init__()  # 调用父类构造函数
        # 构建U-Net结构，从内层到外层逐层嵌套
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # 最内层
        for i in range(num_downs - 5):  # 添加中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 逐步减少滤波器数量
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # 最外层

    # 前向传播函数
    def forward(self, input):
        return self.model(input)  # 通过模型处理输入并返回结果
class UnetSkipConnectionBlock(nn.Module):
    # 构造函数，定义块的配置
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()  # 调用父类构造函数
        self.outermost = outermost  # 是否为最外层
        # 判断是否使用偏置
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc  # 如果没有指定输入通道数，则设为外层通道数
        # 定义向下的卷积操作
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # 定义向上的卷积操作
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # 根据层的位置（最外层、最内层或中间层）配置不同的模块组合
        if outermost:  # 最外层，输出层使用Tanh激活
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 最内层，没有跳跃连接
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # 中间层，包含跳跃连接
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:  # 如果使用dropout
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)  # 将所有层组合成一个序列模型

    # 前向传播函数
    def forward(self, x):
        if self.outermost:  # 如果是最外层，直接返回结果
            return self.model(x)
        else:  # 否则，加上输入（实现跳跃连接）
            return torch.cat([x, self.model(x)], 1)  # 在通道维度上连接输入和输出，实现U-Net的典型特性

def tensor2im(input_tensor, imtype=np.uint8):
    image_tensor = input_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()  # 变成numpy
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # 后处理：变换和缩放
    output_image = image_numpy.astype(imtype)
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image # 返回numpy

def im2tensor(input_image):
    input_image = input_image.convert('RGB')
    transform_list = [
        transforms.Resize([512,512], InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)

    output_tensor = transform(input_image)
    output_tensor = torch.unsqueeze(output_tensor, 0)

    return output_tensor
def cubic_spline_interpolation(points, num=10):
    if len(points) < 4:
        return []
    else:
        points = points
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        # 使用三次样条插值
        tck, u = splprep([x, y], k=3, s=0)
        u_new = np.linspace(0, 1, num=len(points) * num)
        x_spline, y_spline = splev(u_new, tck)

        return [(int(round(x)), int(round(y))) for x, y in zip(x_spline, y_spline)]

def resize_image(img, max_size=1024):
    width, height = img.size
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img
def cv2_preprocess(image):
    # 读取图像文件
    # 如果输入是PIL图像，则转换为NumPy数组
    image_np = np.array(image)
    # 将RGB转换为BGR格式，因为OpenCV使用BGR格式
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 将图像转换为灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对灰度图进行二值化处理
    _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # 创建一个3x3的全1结构元素
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    # 对二值图像进行膨胀操作，迭代两次
    image_binary = cv2.dilate(image_binary, kernel, iterations=2)
    # 对膨胀后的图像进行腐蚀操作，恢复图像形态
    image_binary = cv2.erode(image_binary, kernel=kernel)
    return image_binary
def optimize(image, min_dist, k , inserted, closed, ismask):

        result = []
        image_binary = cv2_preprocess(image)
        # 寻找轮廓
        # contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 遍历找到的轮廓
        for contour in contours:
            # 如果轮廓面积小于或等于20，忽略此轮廓
            if cv2.contourArea(contour) <= 20:
                continue

            # 对轮廓点进行多边形逼近
            contour = cv2.approxPolyDP(contour, 3, closed)
            # 对轮廓进行贝塞尔曲线处理
            contour = bezier( contour=contour, k=k, inserted=inserted,closed=closed)  # 默认值是k=0.4，inserted=100 比较密集

            # 对轮廓进行稀疏处理，减少点的数量
            sparse_contour = list()
            pre_idx = 0
            post_idx = 1
            sparse_contour.append(contour[pre_idx])
            while post_idx < len(contour):
                dist = math.sqrt((contour[pre_idx][0][0] - contour[post_idx][0][0]) ** 2 +
                                 (contour[pre_idx][0][1] - contour[post_idx][0][1]) ** 2)
                if dist >= min_dist:
                    sparse_contour.append(contour[post_idx])
                    pre_idx = post_idx
                    post_idx += 1
                else:
                    post_idx += 1

            contour = np.array(sparse_contour)

            # 再次检查轮廓面积，如果处理后面积仍小于或等于20，忽略此轮廓
            if cv2.contourArea(contour) <= 20:
                continue

            result.append([point[0] for point in contour.tolist()])
        if ismask:
            if len(result) > 1:            #只取最长的边
                # 找出最长的元素
                longest = max(result, key=len )
                # 更新result为只包含最长的那个元素
                result = [longest]
        return result
def total_length(points_list):
    return sum(sqrt((points_list[i][0] - points_list[i-1][0]) ** 2 + (points_list[i][1] - points_list[i-1][1]) ** 2) for i in range(1, len(points_list)))

def pre_process(name,mask,ismask):
    if name == 'mask':
        min_dist, k, inserted = 100, 0.2, 1
    else:
        min_dist , k , inserted=1,0.4,100
    if name == 'ZW':
        mask = cv2_preprocess(mask)
        mask = Image.fromarray(mask)
    else:
        points_data = optimize(mask, min_dist=min_dist, k=k, inserted=inserted, closed=True, ismask=ismask)
        # 创建黑色背景的画布
        canvas = Image.new('RGB', input_image.size, color='black')
        draw = ImageDraw.Draw(canvas)
        # 将点连成曲线并填充内部为白色
        if name == 'PZ+DL':
            points_data = sorted(points_data, key=total_length, reverse=True)
            seen = []
            for index, points_list in enumerate(points_data):
                if points_list not in seen:
                    seen.append(points_list)
                    if len(points_list)>=4:
                        if index == 0:
                            fill = 'white'
                        else:
                            fill = 'black'
                        smoothed_points = cubic_spline_interpolation(points_list)
                        draw.polygon(smoothed_points, fill=fill, outline=None)
        else:
            for points_list in points_data:
                if len(points_list)>=4:
                    smoothed_points = cubic_spline_interpolation(points_list)  # 使用样条插值获取平滑的曲线
                    draw.polygon(smoothed_points, fill='white', outline='white')  # 绘制曲线并填充内部为白色
        mask = canvas.resize(input_image.size)
    return mask

import time

if __name__ == '__main__':

    start_time = time.time()
    model_paths = [
        'mask.pth',
        'PZ+DL.pth',
        'PZ.pth',
        'ST.pth',
        'GZW.pth',
        'ZW.pth'
    ]
    models = {path : generator(gpu_ids=0, ngf=64, init_gain=0.02, load_path=path) for path in model_paths}

    in_dir = 'masterplans'
    out_dir = 'layouts'

    os.makedirs(out_dir,exist_ok=True)
    a = os.listdir(in_dir)
    fff = 0
    for filename in a:
        start_time_per = time.time()
        fff+=1
        input_image_path = os.path.join(in_dir,filename)
        output_image_path = os.path.join(out_dir,filename)
        print('processing: ',input_image_path)

        # 打开输入图
        # input_image_path = '000218.png'
        input_image = Image.open(input_image_path)
        input_image = resize_image(input_image)

        # 生成mask
        model = models['mask.pth']
        input_tensor = im2tensor(input_image)
        output_tensor = model.netG(input_tensor)
        external_mask = tensor2im(output_tensor).resize(input_image.size)

        # 优化mask
        external_mask_optimize = pre_process('mask', external_mask,True)

        # 合并mask和输入图像
        black_image = Image.new("RGB", input_image.size, (0, 0, 0))
        mask = external_mask_optimize.convert('L')
        cover_external_mask = Image.composite(input_image, black_image, mask).resize(input_image.size)

        # 循环生成元素
        landuse_dict = {'mask': external_mask_optimize}
        element_names = [
                         'ST',
                         'PZ+DL',
                         'PZ',
                         'GZW',
                         'ZW'
                         ]
        for name in element_names:
            model = models[f'{name}.pth']
            input_tensor = im2tensor(cover_external_mask)
            output_tensor = model.netG(input_tensor)
            mask = tensor2im(output_tensor).resize(input_image.size)
            mask = pre_process(name, mask,False)
            landuse_dict[name] = mask

        # 赋色
        canvas = Image.new('RGBA', input_image.size, color=(0, 0, 0, 255))
        colors = {
            'mask': [0, 245, 0],
            'GZW': [245, 0, 245],
            # 'LD': [0, 245, 0],
            'PZ': [245, 245, 0],
            'PZ+DL': [231, 135, 63],
            'ST': [0, 245, 245],
            'ZW': [0, 142, 57],
        }
        for name in landuse_dict:
            landuse = landuse_dict[name]
            landuse_gray = landuse.convert("L")  # 将黑白图像转换为灰度图像
            color = tuple(colors[name])  # 获取对应颜色的RGB值
            colored_landuse = ImageOps.colorize(landuse_gray, black='black', white=color).convert("RGBA")  # 使用ImageOps.colorize对灰度图像进行颜色填充
            mask = landuse_gray.point(lambda p: 255 if p > 128 else 0).convert("L")  # 创建一个掩码，仅选择白色区域
            canvas.paste(colored_landuse, (0, 0), mask)  # 将赋色后的图像依次黏贴到canvas上，只叠加有颜色的部分

        # canvas = canvas.convert('RGB')

        canvas.save(output_image_path.replace('.jpg','.png'),'PNG')
        print('done: ',output_image_path)
        print('#unprocess:', 9286-fff)
        print('per_time: ', f"{time.time() - start_time_per:.2f} s")
        print('time: ', f"{time.time() - start_time:.2f} s")
        print('='*20,fff,"="*20)

    #########################################################