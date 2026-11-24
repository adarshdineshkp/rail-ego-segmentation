"""PIDNet (Xu et al., CVPR 2023), re-implemented so that module and parameter
names match the official repo (https://github.com/XuJiacong/PIDNet).
That means the official ImageNet / Cityscapes checkpoints load directly.

Output in training mode (augment=True): [P-branch logits, main logits, boundary logit]
Output with augment=False: main logits only. All outputs are at 1/8 input resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOM = 0.1
ALIGN_CORNERS = False


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)


# --------------------------------------------------------------------------- blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOM)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        return self.conv2(self.relu(self.bn2(x)))


def _bn_relu_conv(cin, cout, k=1, groups=1):
    return nn.Sequential(
        nn.BatchNorm2d(cin, momentum=BN_MOM),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
    )


def _pool_branch(pool, cin, cout):
    return nn.Sequential(
        pool,
        nn.BatchNorm2d(cin, momentum=BN_MOM),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
    )


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = _pool_branch(nn.AvgPool2d(5, 2, 2), inplanes, branch_planes)
        self.scale2 = _pool_branch(nn.AvgPool2d(9, 4, 4), inplanes, branch_planes)
        self.scale3 = _pool_branch(nn.AvgPool2d(17, 8, 8), inplanes, branch_planes)
        self.scale4 = _pool_branch(nn.AdaptiveAvgPool2d((1, 1)), inplanes, branch_planes)
        self.scale0 = _bn_relu_conv(inplanes, branch_planes, 1)
        self.process1 = _bn_relu_conv(branch_planes, branch_planes, 3)
        self.process2 = _bn_relu_conv(branch_planes, branch_planes, 3)
        self.process3 = _bn_relu_conv(branch_planes, branch_planes, 3)
        self.process4 = _bn_relu_conv(branch_planes, branch_planes, 3)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes, 1)
        self.shortcut = _bn_relu_conv(inplanes, outplanes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        xs = [self.scale0(x)]
        xs.append(self.process1(_up(self.scale1(x), size) + xs[0]))
        xs.append(self.process2(_up(self.scale2(x), size) + xs[1]))
        xs.append(self.process3(_up(self.scale3(x), size) + xs[2]))
        xs.append(self.process4(_up(self.scale4(x), size) + xs[3]))
        return self.compression(torch.cat(xs, 1)) + self.shortcut(x)


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = _pool_branch(nn.AvgPool2d(5, 2, 2), inplanes, branch_planes)
        self.scale2 = _pool_branch(nn.AvgPool2d(9, 4, 4), inplanes, branch_planes)
        self.scale3 = _pool_branch(nn.AvgPool2d(17, 8, 8), inplanes, branch_planes)
        self.scale4 = _pool_branch(nn.AdaptiveAvgPool2d((1, 1)), inplanes, branch_planes)
        self.scale0 = _bn_relu_conv(inplanes, branch_planes, 1)
        self.scale_process = _bn_relu_conv(branch_planes * 4, branch_planes * 4, 3, groups=4)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes, 1)
        self.shortcut = _bn_relu_conv(inplanes, outplanes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x_ = self.scale0(x)
        scales = [_up(s(x), size) + x_ for s in (self.scale1, self.scale2, self.scale3, self.scale4)]
        scale_out = self.scale_process(torch.cat(scales, 1))
        return self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)


class PagFM(nn.Module):
    """Pixel-attention-guided fusion of the I branch (y) into the P branch (x)."""

    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                 nn.BatchNorm2d(mid_channels, momentum=BN_MOM))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                 nn.BatchNorm2d(mid_channels, momentum=BN_MOM))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, 1, bias=False),
                                    nn.BatchNorm2d(in_channels, momentum=BN_MOM))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        size = x.shape[-2:]
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = _up(self.f_y(y), size)
        x_k = self.f_x(x)
        if self.with_channel:
            sim = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = _up(y, size)
        return (1 - sim) * x + sim * y


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels, momentum=BN_MOM))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels, momentum=BN_MOM))

    def forward(self, p, i, d):
        att = torch.sigmoid(d)
        return self.conv_p((1 - att) * i + p) + self.conv_i(i + att * p)


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = _bn_relu_conv(in_channels, out_channels, 3)

    def forward(self, p, i, d):
        att = torch.sigmoid(d)
        return self.conv(att * p + (1 - att) * i)


# --------------------------------------------------------------------------- network
class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128,
                 augment=True):
        super().__init__()
        self.augment = augment

        # I branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=BN_MOM), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=BN_MOM), nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P branch
        self.compression3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
                                          nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
        self.compression4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
                                          nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes, momentum=BN_MOM))
            self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.diff4 = nn.Sequential(nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # heads (aux heads are always built so checkpoints load the same way)
        self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
        self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    @staticmethod
    def _downsample(block, inplanes, planes, stride):
        if stride != 1 or inplanes != planes * block.expansion:
            return nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM),
            )
        return None

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = [block(inplanes, planes, stride, self._downsample(block, inplanes, planes, stride))]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        return block(inplanes, planes, stride, self._downsample(block, inplanes, planes, stride),
                     no_relu=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        size = x_.shape[-2:]  # 1/8 resolution (P/D branches)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + _up(self.diff3(x), size)
        temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + _up(self.diff4(x), size)
        temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = _up(self.spp(self.layer5(x)), size)
        out = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            return [self.seghead_p(temp_p), out, self.seghead_d(temp_d)]
        return out


VARIANTS = {
    "pidnet_s": dict(m=2, n=3, planes=32, ppm_planes=96, head_planes=128),
    "pidnet_m": dict(m=2, n=3, planes=64, ppm_planes=96, head_planes=128),
    "pidnet_l": dict(m=3, n=4, planes=64, ppm_planes=112, head_planes=256),
}

# classifier convs that depend on the label set; skipped when fine-tuning to new classes
HEAD_KEYS = ("final_layer.conv2.", "seghead_p.conv2.")


def build_model(name, num_classes, augment=True):
    name = name.lower()
    if name not in VARIANTS:
        raise ValueError(f"Unknown model {name!r}; choose one of {list(VARIANTS)}")
    return PIDNet(num_classes=num_classes, augment=augment, **VARIANTS[name])


def load_pretrained(model, path, skip_head=True, log=print):
    """Load official PIDNet ImageNet/Cityscapes weights (or our own checkpoints).
    Only tensors whose name and shape match are copied. Returns (n_loaded, n_total)."""
    from railseg.utils import safe_torch_load

    ckpt = safe_torch_load(path)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    own = model.state_dict()
    loaded, skipped_shape = {}, []
    for k, v in ckpt.items():
        while k.startswith(("module.", "model.")):
            k = k.split(".", 1)[1]
        if skip_head and k.startswith(HEAD_KEYS):
            continue
        if k in own:
            if own[k].shape == v.shape:
                loaded[k] = v
            else:
                skipped_shape.append(k)
    own.update(loaded)
    model.load_state_dict(own)
    n_params = sum(1 for k in own if not k.endswith("num_batches_tracked"))
    n_loaded = sum(1 for k in loaded if not k.endswith("num_batches_tracked"))
    log(f"Pretrained: loaded {n_loaded}/{n_params} tensors from {path}"
        + (f" ({len(skipped_shape)} skipped for shape mismatch)" if skipped_shape else "")
        + (" [classifier heads re-initialised]" if skip_head else ""))
    return n_loaded, n_params
