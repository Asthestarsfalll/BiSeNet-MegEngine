import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.module import BatchNorm2d
from megengine import hub
from .resnet import Resnet18


class BaseModel(M.Module):
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, M.Conv2d):
                M.init.msra_normal_(ly.weight, a=1.)
                if not ly.bias is None:
                    M.init.zeros_(ly.bias)


class ConvBNReLU(BaseModel):

    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = M.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = M.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSample(M.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = M.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = M.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        M.init.xavier_normal_(self.proj.weight)


class Upsample(M.Module):
    def __init__(self, scale_factor, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.kargs = {
            "scale_factor": scale_factor,
            "mode": mode,
            "align_corners": align_corners
        }

    def forward(self, x):
        return F.nn.interpolate(x, **self.kargs)


class BiSeNetOutput(BaseModel):
    def __init__(
        self,
        in_chan,
        mid_chan,
        n_classes,
        up_factor=32
    ):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = M.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = Upsample(scale_factor=up_factor,
                           mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x


class AttentionRefinementModule(BaseModel):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = M.Conv2d(
            out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.mean(feat, axis=(2, 3), keepdims=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = F.sigmoid(atten)
        out = F.mul(feat, atten)
        return out


class ContextPath(BaseModel):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = Upsample(scale_factor=2.)
        self.up16 = Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = F.mean(feat32, axis=(2, 3), keepdims=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16


class SpatialPath(BaseModel):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat


class FeatureFusionModule(BaseModel):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = M.Conv2d(
            out_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn = M.BatchNorm2d(out_chan)

        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = F.concat([fsp, fcp], axis=1)
        feat = self.convblk(fcat)
        atten = F.mean(feat, axis=(2, 3), keepdims=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = F.sigmoid(atten)
        feat_atten = F.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNetV1(BaseModel):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.shape[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/119/files/7a1d207e-cf91-420f-aeb5-770d3a87b901"
)
def bisenetv1(**kwargs):
    return BiSeNetV1(n_classes=19, **kwargs)