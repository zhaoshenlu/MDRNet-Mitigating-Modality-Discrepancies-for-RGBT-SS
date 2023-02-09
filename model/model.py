import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F

class TFNet(nn.Module):##### structure of MDRNet+ #####
    def __init__(self, n_class):
        super(TFNet, self).__init__()
        model1 = models.resnet50(pretrained=True)
        model3 = models.resnet50(pretrained=True)
        model4 = models.resnet50(pretrained=True)
        model6 = models.resnet50(pretrained=True)
        model7 = models.resnet50(pretrained=True)
###########################  T encoder  ###########################
        self.encoder_T3_conv1 = model3.conv1
        self.encoder_T3_bn1 = model3.bn1
        self.encoder_T3_relu = model3.relu
        self.encoder_T3_maxpool = model3.maxpool
        self.encoder_T3_layer1 = model3.layer1
        self.encoder_T3_layer2 = model3.layer2
        self.encoder_T3_layer3 = model3.layer3
        self.encoder_T3_layer4 = model3.layer4
###########################  RGB encoder  ##########################
        self.encoder_RGB3_conv1 = model6.conv1
        self.encoder_RGB3_bn1 = model6.bn1
        self.encoder_RGB3_relu = model6.relu
        self.encoder_RGB3_maxpool = model6.maxpool
        self.encoder_RGB3_layer1 = model6.layer1
        self.encoder_RGB3_layer2 = model6.layer2
        self.encoder_RGB3_layer3 = model6.layer3
        self.encoder_RGB3_layer4 = model6.layer4
##################  pseudo and real T encoder （Siamese） ##################
        self.encoder_fakeT_conv1 = model1.conv1
        self.encoder_fakeT_bn1 = model1.bn1
        self.encoder_fakeT_relu = model1.relu
        self.encoder_fakeT_maxpool = model1.maxpool
        self.encoder_fakeT_layer1 = model1.layer1
        self.encoder_fakeT_layer2 = model1.layer2
        self.encoder_fakeT_layer3 = model1.layer3
        self.encoder_fakeT_layer4 = model1.layer4
##################  pseudo and real RGB encoder （Siamese） #################
        self.encoder_fakeRGB_conv1 = model4.conv1
        self.encoder_fakeRGB_bn1 = model4.bn1
        self.encoder_fakeRGB_relu = model4.relu
        self.encoder_fakeRGB_maxpool = model4.maxpool
        self.encoder_fakeRGB_layer1 = model4.layer1
        self.encoder_fakeRGB_layer2 = model4.layer2
        self.encoder_fakeRGB_layer3 = model4.layer3
        self.encoder_fakeRGB_layer4 = model4.layer4
#################  translation network  ################
        self.fake_T_decoder1 = TransConvBnLeakyRelu2d(2048, 1024)
        self.fake_T_decoder2 = ConvBnrelu2d_3(1024, 1024)
        self.fake_T_decoder3 = TransConvBnLeakyRelu2d(1024, 512)
        self.fake_T_decoder4 = ConvBnrelu2d_3(512, 512)
        self.fake_T_decoder5 = TransConvBnLeakyRelu2d(512, 256)
        self.fake_T_decoder6 = ConvBnrelu2d_3(256, 256)
        self.fake_T_decoder7 = TransConvBnLeakyRelu2d(256, 64)
        self.fake_T_decoder8 = ConvBnrelu2d_3(64, 64)
        self.fake_T_decoder9 = TransConvBnLeakyRelu2d(64, 64)
        self.fake_T_last = Conv_3(64, 3)
        
        self.fake_RGB_decoder1 = TransConvBnLeakyRelu2d(2048, 1024)
        self.fake_RGB_decoder2 = ConvBnrelu2d_3(1024, 1024)
        self.fake_RGB_decoder3 = TransConvBnLeakyRelu2d(1024, 512)
        self.fake_RGB_decoder4 = ConvBnrelu2d_3(512, 512)
        self.fake_RGB_decoder5 = TransConvBnLeakyRelu2d(512, 256)
        self.fake_RGB_decoder6 = ConvBnrelu2d_3(256, 256)
        self.fake_RGB_decoder7 = TransConvBnLeakyRelu2d(256, 64)
        self.fake_RGB_decoder8 = ConvBnrelu2d_3(64, 64)
        self.fake_RGB_decoder9 = TransConvBnLeakyRelu2d(64, 64)
        self.fake_RGB_last = Conv_3(64, 3)
############################  reconstruction T  ################
        self.toclassconv1 = TransConvBnLeakyRelu2d(2048, 1024)
        #self.toclassconv2 = ConvBnrelu2d_3(1024, 1024)
        self.toclassconv3 = TransConvBnLeakyRelu2d(1024, 512)
        #self.toclassconv4 = ConvBnrelu2d_3(512, 512)
        self.toclassconv5 = TransConvBnLeakyRelu2d(512, 256)
        #self.toclassconv6 = ConvBnrelu2d_3(256, 256)
        self.toclassconv7 = TransConvBnLeakyRelu2d(256, 64)
        #self.toclassconv8 = ConvBnrelu2d_3(64, 64)
        self.toclassconv9 = TransConvBnLeakyRelu2d(64, 64)
        self.latent_t = ConvBnrelu2d_3(64, 64)
        self.recon_t = Conv_3(64, 3)
##########################  reconstruction RGB  #######################
        self.toclassconv10 = TransConvBnLeakyRelu2d(2048, 1024)
        #self.toclassconv11 = ConvBnrelu2d_3(1024, 1024)
        self.toclassconv12 = TransConvBnLeakyRelu2d(1024, 512)
        #self.toclassconv13 = ConvBnrelu2d_3(512, 512)
        self.toclassconv14 = TransConvBnLeakyRelu2d(512, 256)
        #self.toclassconv15 = ConvBnrelu2d_3(256, 256)
        self.toclassconv16 = TransConvBnLeakyRelu2d(256, 64)
        #self.toclassconv17 = ConvBnrelu2d_3(64, 64)
        self.toclassconv18 = TransConvBnLeakyRelu2d(64, 64)
        self.latent_rgb = ConvBnrelu2d_3(64, 64)
        self.recon_rgb = Conv_3(64, 3)
###########################################################  fusion encoder   #######################################################################
        self.encoder_fusion_maxpool = model7.maxpool
        self.encoder_fusion_layer1 = model7.layer1
        self.encoder_fusion_layer2 = model7.layer2
        self.encoder_fusion_layer3 = Spatial_ASPP(512, 256)
        self.encoder_fusion_layer4 = model7.layer3
        self.encoder_fusion_layer5 = ASPP_Global(1024, 512)
#####################################  CWF weight s prediction  ############################################
#        self.s_conv = ConvBnrelu2d_3(1024, 1024)
        self.s_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.s_weight_classifier = nn.Sequential(            
            Convrelu_1(2048, 1024),
            Conv_3(1024, 1024),
            nn.Sigmoid()
            )
#####################################  CWF weight d3 prediction  ############################################
#        self.d3_conv = ConvBnrelu2d_3(512, 512)
        self.d3_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.d3_weight_classifier = nn.Sequential(            
            Convrelu_1(1024, 512),
            Conv_3(512, 512),
            nn.Sigmoid()
            )  
#####################################  CWF weight d2 prediction  ############################################
#        self.d2_conv = ConvBnrelu2d_3(256, 256)
        self.d2_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.d2_weight_classifier = nn.Sequential(            
            Convrelu_1(512, 256),
            Conv_3(256, 256),
            nn.Sigmoid()
            )                         
#####################################  CWF weight d1 prediction  ############################################
#        self.d1_conv = ConvBnrelu2d_3(64, 64)
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.d1_weight_classifier = nn.Sequential(            
            Convrelu_1(128, 64),
            Conv_3(64, 64),
            nn.Sigmoid()
            )                   
###########################################################  fusion decoder #########################################################################    
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(1024,512)
        self.fusion_conv1 = ConvBnrelu2d_3(512, 512)
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(512,256)       
        self.fusion_conv2 = ConvBnrelu2d_3(256, 256)
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(256,64)       
        self.fusion_conv3 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(64,64)    
        self.fusion_conv4 = nn.Conv2d(64, n_class, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv4.weight.data)
        
    def forward(self, input_rgb, input_t):
        rgb1 = input_rgb
        rgb2 = input_rgb
        t1 = input_t
        t2 = input_t

        rgb2 = self.encoder_RGB3_conv1(rgb2)
        rgb2 = self.encoder_RGB3_bn1(rgb2)
        rgb2 = self.encoder_RGB3_relu(rgb2)
        fusion1 = rgb2#1/2
        rgb2 = self.encoder_RGB3_maxpool(rgb2)           
        rgb2 = self.encoder_RGB3_layer1(rgb2)  
        fusion2 = rgb2#1/4
        rgb2 = self.encoder_RGB3_layer2(rgb2)
        fusion3 = rgb2#1/8
        rgb2 = self.encoder_RGB3_layer3(rgb2)
        fusion0 = rgb2#1/16
        rgb2 = self.encoder_RGB3_layer4(rgb2)
        rgb2 = self.fake_RGB_decoder1(rgb2)
        rgb2 = self.fake_RGB_decoder2(rgb2 + fusion0)
        rgb2 = self.fake_RGB_decoder3(rgb2)
        rgb2 = self.fake_RGB_decoder4(rgb2 + fusion3)
        rgb2 = self.fake_RGB_decoder5(rgb2)
        rgb2 = self.fake_RGB_decoder6(rgb2 + fusion2)
        rgb2 = self.fake_RGB_decoder7(rgb2)
        rgb2 = self.fake_RGB_decoder8(rgb2 + fusion1)
        rgb2 = self.fake_RGB_decoder9(rgb2)
        rgb2 = self.fake_RGB_last(rgb2)
        rgb2 = torch.sigmoid(rgb2)
        fake_T = rgb2
        
        t1 = self.encoder_fakeT_conv1(t1)
        t1 = self.encoder_fakeT_bn1(t1)
        t1 = self.encoder_fakeT_relu(t1)
        ff = self.encoder_fakeT_conv1(fake_T)
        ff = self.encoder_fakeT_bn1(ff)
        ff = self.encoder_fakeT_relu(ff)
        x1 = t1
        x2 = ff
        
        t1 = self.encoder_fakeT_maxpool(t1)
        t1 = self.encoder_fakeT_layer1(t1)
        ff = self.encoder_fakeT_maxpool(ff)
        ff = self.encoder_fakeT_layer1(ff)
        x3 = t1
        x4 = ff
        
        t1 = self.encoder_fakeT_layer2(t1)
        ff = self.encoder_fakeT_layer2(ff)
        x5 = t1
        x6 = ff
        
        t1 = self.encoder_fakeT_layer3(t1)
        ff = self.encoder_fakeT_layer3(ff)
        x7 = t1
        x8 = ff
        
        t1 = self.encoder_fakeT_layer4(t1)
        ff = self.encoder_fakeT_layer4(ff)
        x9 = t1
        x10 = ff
        
        t1 = self.toclassconv1(t1)
        #t1 = self.toclassconv2(t1 + x7)
        t1 = self.toclassconv3(t1)
        #t1 = self.toclassconv4(t1 + x5)
        t1 = self.toclassconv5(t1)
        #t1 = self.toclassconv6(t1 + x3)
        t1 = self.toclassconv7(t1)
        #t1 = self.toclassconv8(t1 + x1)
        t1 = self.toclassconv9(t1)
        t1 = self.latent_t(t1)
        t1 = self.recon_t(t1)
        #t1 = torch.sigmoid(t1)
        T_recon = t1

        t2 = self.encoder_T3_conv1(t2)
        t2 = self.encoder_T3_bn1(t2)
        t2 = self.encoder_T3_relu(t2)
        fusion4 = t2#1/2
        t2 = self.encoder_T3_maxpool(t2)
        t2 = self.encoder_T3_layer1(t2)
        fusion5 = t2#1/4
        t2 = self.encoder_T3_layer2(t2)
        fusion6 = t2#1/8
        t2 = self.encoder_T3_layer3(t2)
        fusion7 = t2#1/16
        t2 = self.encoder_T3_layer4(t2)
        t2 = self.fake_T_decoder1(t2)
        t2 = self.fake_T_decoder2(t2 + fusion7)
        t2 = self.fake_T_decoder3(t2)
        t2 = self.fake_T_decoder4(t2 + fusion6)
        t2 = self.fake_T_decoder5(t2)
        t2 = self.fake_T_decoder6(t2 + fusion5)
        t2 = self.fake_T_decoder7(t2)
        t2 = self.fake_T_decoder8(t2 + fusion4)
        t2 = self.fake_T_decoder9(t2)
        t2 = self.fake_T_last(t2)
        t2 = torch.sigmoid(t2)
        fake_RGB = t2
        
        rgb1 = self.encoder_fakeRGB_conv1(rgb1)
        rgb1 = self.encoder_fakeRGB_bn1(rgb1)
        rgb1 = self.encoder_fakeRGB_relu(rgb1)
        mm = self.encoder_fakeRGB_conv1(fake_RGB)
        mm = self.encoder_fakeRGB_bn1(mm)
        mm = self.encoder_fakeRGB_relu(mm)
        x11 = rgb1
        x12 = mm
        
        rgb1 = self.encoder_fakeRGB_maxpool(rgb1)
        rgb1 = self.encoder_fakeRGB_layer1(rgb1)
        mm = self.encoder_fakeRGB_maxpool(mm)
        mm = self.encoder_fakeRGB_layer1(mm)
        x13 = rgb1
        x14 = mm

        rgb1 = self.encoder_fakeRGB_layer2(rgb1)
        mm = self.encoder_fakeRGB_layer2(mm)
        x15 = rgb1
        x16 = mm
        
        rgb1 = self.encoder_fakeRGB_layer3(rgb1)
        mm = self.encoder_fakeRGB_layer3(mm)
        x17 = rgb1
        x18 = mm
        
        rgb1 = self.encoder_fakeRGB_layer4(rgb1)
        mm = self.encoder_fakeRGB_layer4(mm)
        x19 = rgb1
        x20 = mm
        
        rgb1 = self.toclassconv10(rgb1)
        #rgb1 = self.toclassconv11(rgb1 + x17)
        rgb1 = self.toclassconv12(rgb1)
        #rgb1 = self.toclassconv13(rgb1 + x15)
        rgb1 = self.toclassconv14(rgb1)
        #rgb1 = self.toclassconv15(rgb1 + x13)
        rgb1 = self.toclassconv16(rgb1)
        #rgb1 = self.toclassconv17(rgb1 + x11)
        rgb1 = self.toclassconv18(rgb1)
        rgb1 = self.latent_rgb(rgb1)
        rgb1 = self.recon_rgb(rgb1)
        #rgb1 = torch.sigmoid(rgb1)
        RGB_recon = rgb1

        d_1 = torch.cat((fusion1, fusion4), dim=1)
        d_2 = torch.cat((fusion2, fusion5), dim=1)
        d_3 = torch.cat((fusion3, fusion6), dim=1)
        seg = torch.cat((fusion0, fusion7), dim=1)
#        d_1 = self.d1_conv(d_1)
#        d_2 = self.d2_conv(d_2)
#        d_3 = self.d3_conv(d_3)
#        seg = self.s_conv(seg)
      
        
        weight_d1 = self.d1_weight_classifier(d_1)
        weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
        w_d1 = weight_d1
        w_d2 = 1-weight_d1 

        
        weight_d2 = self.d2_weight_classifier(d_2)
        weight_d2 = self.d2_weight_classifier_avgpool(weight_d2)
        w_d3 = weight_d2
        w_d4 = 1-weight_d2
        
        
        weight_d3 = self.d3_weight_classifier(d_3)
        weight_d3 = self.d3_weight_classifier_avgpool(weight_d3)
        w_d5 = weight_d3
        w_d6 = 1-weight_d3 

        
        weight_seg = self.s_weight_classifier(seg)   
        weight_seg = self.s_weight_classifier_avgpool(weight_seg)        
        w_s1 = weight_seg
        w_s2 = 1-weight_seg
        
        
        fusion = w_d1*fusion1 + w_d2*fusion4 
        m1 = fusion        
        fusion = self.encoder_fusion_maxpool(fusion)
        fusion = self.encoder_fusion_layer1(fusion)                       
        fusion = fusion + w_d3*fusion2 + w_d4*fusion5
        m2 = fusion
        fusion = self.encoder_fusion_layer2(fusion)
        fusion = fusion + w_d5*fusion3 + w_d6*fusion6 
        
        
        fusion = self.encoder_fusion_layer3(fusion)
        
        m3 = fusion
        
        fusion = self.encoder_fusion_layer4(fusion)
        
        fusion = fusion + w_s1*fusion0 + w_s2*fusion7
        before = fusion
        fusion = self.encoder_fusion_layer5(fusion)
        after = fusion
        
        fusion = self.fusion_decoder0(fusion)
                
        fusion = fusion + m3 
        fusion = self.fusion_conv1(fusion)
        fusion = self.fusion_decoder1(fusion)

        fusion = fusion + m2
        fusion = self.fusion_conv2(fusion)
        fusion = self.fusion_decoder2(fusion)

        fusion = fusion + m1
        fusion = self.fusion_conv3(fusion)
        fusion = self.fusion_decoder3(fusion)
        fusion = self.fusion_conv4(fusion)
        return fusion, fake_T, fake_RGB, T_recon, RGB_recon, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20

class ConvBnrelu2d_3(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvBnrelu2d_1(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class DilationConvBn2d_3(nn.Module):
    # Dilationconvolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, dilation, padding, kernel_size=3, stride=1, groups=1, bias=False):
        super(DilationConvBn2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ASPP_Global(nn.Module):   ##################  MCC  ##################
    def __init__(self, in_channels, out_channels):
        super(ASPP_Global, self).__init__()
        
        self.conv1 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv1 = DilationConvBn2d_3(out_channels, out_channels, dilation=1, padding=1)
        
        self.conv2 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv2 = DilationConvBn2d_3(out_channels, out_channels, dilation=6, padding=6)
        
        self.conv3 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv3 = DilationConvBn2d_3(out_channels, out_channels, dilation=12, padding=12)
        
        self.conv4 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv4 = DilationConvBn2d_3(out_channels, out_channels, dilation=18, padding=18)
        
        self.conv5 = ConvBnrelu2d_1(in_channels*2, in_channels)
        self.conv6 = ConvBnrelu2d_1(in_channels, in_channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_0 = x        
        x_1 = self.Dconv1(self.conv1(x))
        x_2 = self.Dconv2(self.conv2(x))
        x_3 = self.Dconv3(self.conv3(x))
        x_4 = self.Dconv4(self.conv4(x))
        x = torch.cat((x_1, x_2), dim=1)        
        x = torch.cat((x, x_3), dim=1)        
        x = torch.cat((x, x_4), dim=1)
        x = self.conv5(x)
        x = x.view(b, c, h*w)
        attention1 = x
        
        attention1 = attention1/(torch.norm(attention1, dim=1, keepdim=True)+1e-5)

        attention1 = F.relu(torch.matmul(attention1, attention1.transpose(1,2)),inplace=True)
        attention1 = attention1/(torch.sum(attention1, dim=1,keepdim=True)+1e-5)
        #print(attention1.shape)
        x_5 = torch.matmul(attention1,x).view(b,-1,h,w)
        
        attention2 = x_0.view(b, c, h*w)
        attention2 = attention2/(torch.norm(attention2, dim=1, keepdim=True)+1e-5)
        #print(attention2.shape)
        attention2 = F.relu(torch.matmul(attention2, attention2.transpose(1,2)),inplace=True)
        #print(attention2.shape)
        attention2 = attention2/(torch.sum(attention2, dim=1,keepdim=True)+1e-5)
        x_6 = torch.matmul(attention2,x).view(b,-1,h,w)
        
        x = self.conv6(x_0+x_5+x_6)
                
        return x
        
class Spatial_ASPP(nn.Module): ###########  MSC  ##########
    def __init__(self, in_channels, out_channels):
        super(Spatial_ASPP, self).__init__()
        self.conv1 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv1 = DilationConvBn2d_3(out_channels, out_channels, dilation=1, padding=1)
        
        self.conv2 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv2 = DilationConvBn2d_3(out_channels, out_channels, dilation=6, padding=6)
        
        self.conv3 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv3 = DilationConvBn2d_3(out_channels, out_channels, dilation=12, padding=12)
        
        self.conv4 = ConvBnrelu2d_1(in_channels, out_channels)
        self.Dconv4 = DilationConvBn2d_3(out_channels, out_channels, dilation=18, padding=18)
        
        self.conv5 = ConvBnrelu2d_1(in_channels*2, in_channels)
#        self.conv6 = ConvBnrelu2d_3(in_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_0 = x        
        x_1 = self.Dconv1(self.conv1(x))
        x_2 = self.Dconv2(self.conv2(x))
        x_3 = self.Dconv3(self.conv3(x))
        x_4 = self.Dconv4(self.conv4(x))
        x = torch.cat((x_1, x_2), dim=1)        
        x = torch.cat((x, x_3), dim=1)        
        x = torch.cat((x, x_4), dim=1)
        x = self.conv5(x)
        x_visual = x
        x = x.view(b, c, h*w)
#        print(x.shape)
        attention1 = x
        attention2 = x_0.view(b, c, h*w)
        
        attention1 = attention1/(torch.norm(attention1, dim=1, keepdim=True)+1e-5)
        attention1 = F.relu(torch.matmul(attention1.transpose(1,2), attention1),inplace=True)
#        print(attention1.shape)
        attention1 = attention1/(torch.sum(attention1, dim=1,keepdim=True)+1e-5)
        x_5 = torch.matmul(x, attention1).view(b,-1,h,w)
        
        attention2 = attention2/(torch.norm(attention2, dim=1, keepdim=True)+1e-5)
        attention2 = F.relu(torch.matmul(attention2.transpose(1,2), attention2),inplace=True)
#        print(attention2.shape)
        attention2 = attention2/(torch.sum(attention2, dim=1,keepdim=True)+1e-5)
        x_6 = torch.matmul(x, attention2).view(b,-1,h,w)
        
        x_vis = x_6
        x = x_0 + x_5 + x_6
        return x
        
class TransConvBnLeakyRelu2d(nn.Module):
    # deconvolution
    # batch normalization
    # Lrelu
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()      
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)  
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)   
                               
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))  
        
class Convrelu_1(nn.Module):
    # convolution
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(Convrelu_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)       
        nn.init.xavier_uniform_(self.conv.weight.data)           
    def forward(self, x):
        return F.relu(self.conv(x))

class Conv_3(nn.Module):
    # convolution
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)        
        nn.init.xavier_uniform_(self.conv.weight.data)
    def forward(self, x):
        return self.conv(x)

class ConvBnleakyrelu2d_3(nn.Module):
    # convolution
    # batch normalization
    # Leakyrelu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnleakyrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, img):
        return self.model(img)
