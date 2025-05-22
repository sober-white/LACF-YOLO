######################################## Modulate_Deformconv start #########################################

class Modulate_Deformconv(nn.Module):
    def __init__(self, in_channels, zero_init_offset=False, norm_type='GN') -> None:
        super().__init__()
        self.with_norm = norm_type is not None
        bias = not self.with_norm
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        if norm_type == 'GN':
            norm_dict = dict(type='GN', num_groups=16, requires_grad=True)
        elif norm_type == 'BN':
            norm_dict = dict(type='BN', requires_grad=True)
        
        self.offset_mask_conv = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self._init_weights()

        self.conv = ModulatedDeformConv2d(
            in_channels, in_channels, 3, stride=1, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_dict, in_channels)[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.offset_mask_conv, 0)
    
        
    def forward(self, x):
        """Forward function."""
        offset_mask = self.offset_mask_conv(x)
        offset = offset_mask[:, :self.offset_dim, :, :]
        mask = offset_mask[:, self.offset_dim:, :, :].sigmoid()
        
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x
    
######################################## Modulate_Deformconv end #########################################

######################################## Modulate_Deformconv_bottleneck start #########################################

class Modulate_Deformconv_bottleneck(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = Modulate_Deformconv(c_)
        self.cv3 = Conv(c_, c2, k[1], 1, g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
    
######################################## Modulate_Deformconv_bottleneck end #########################################

######################################## ADM start #######################################

class C2f_Modulate_Deformconv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Modulate_Deformconv_bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## ADM end #########################################

######################################## Contextfusion start #######################################
    
class Contextfusion(nn.Module):
    def __init__(self, c1, c2, r=4) -> None:
        super().__init__()
        # c1, c2 = c[0], c[1]
        inter_channels = int(c2 // r) 

        if c1==c2:
            self.adjust_conv = nn.Identity() 
        else:
            self.adjust_conv = Conv(c1, c2, 1)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(2 * c2, 2 * inter_channels),
            Conv(2 * inter_channels, 2 * c2, act=nn.Sigmoid()),
        )
        self.local_att = nn.Sequential(
            Conv(2 * c2, inter_channels, 1),
            Conv(inter_channels, 2 * c2, 1, act=False),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(2 * c2, inter_channels, 1),
            Conv(inter_channels, 2 * c2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        x1 = self.adjust_conv(x1)
        input = torch.cat([x1, x2], dim=1) #(b, 2c2, h, w)
        recal_w = self.Recalibrate(input) #(b, 2c2, h, w)
        recal_input = recal_w * input
        recal_input = recal_input + input #(b, 2c2, h, w)
        local_w = self.local_att(recal_input)  ##spatial attention
        global_w = self.global_att(recal_input) ## channel attention
        x = self.sigmoid(local_w * global_w) 
        x1_att, x2_att = torch.split(x, [x2.size()[1], x2.size()[1]], 1)
        x1_att = x1 * x1_att
        x2_att = x2 * x2_att
        out = torch.cat([x1+x2_att, x2+x1_att], 1)
        return out
    
######################################## Contextfusion end #########################################
