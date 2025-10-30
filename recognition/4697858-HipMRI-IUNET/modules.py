from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as Functional


class ConvBlock(nn.Module):
    """
    Double convolution block, going from Conv2d to InstanceNorm to LeakyReLu, twice

    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels
        p_drop: Probability of dropout
    """
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channel, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channel, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        if p_drop > 0:
            self.dropout = nn.Dropout2d(p_drop)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.act1(self.in1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.in2(self.conv2(x)))
        return x


class DownSample(nn.Module):
    """
    Downsampling block with a strided convolution, followed by a ConvBlock

    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels
        p_drop: Probability of dropout
    """
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        # Strided convolution block
        self.down = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=2, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        # ConvBlock
        self.block = ConvBlock(out_channel, out_channel, p_drop)

    def forward(self, x):
        x = self.act(self.norm(self.down(x)))
        x = self.block(x)
        return x


class UpSample(nn.Module):
    """
    Upsampling block with a strided transpose convolution block followed by a ConvBlock
    """
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_channel, out_channel, p_drop)

    def forward(self, x, skip):
        x = self.up(x)

        # Interpolates and handles size mismatches
        if (x.shape[-2], x.shape[-1]) != (skip.shape[-2], skip.shape[-1]):
            x = Functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


class IUNet2D(nn.Module):
    """
    Improved 2D UNet for segmentation

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channel: Number of filters at base
        depth: Number of down and up sampling layers
        p_drop: Dropout probability
    """
    def __init__(self, in_channels:int=1, out_channels:int=1, base_channel:int=32, depth:int=4, p_drop:float=0.0):
        super().__init__()

        channels = []
        for i in range(depth):
            channels.append(base_channel * (2**i))

        # Encoder block
        self.encoder0 = ConvBlock(in_channels, channels[0], p_drop)

        self.downs = nn.ModuleList()
        # Produce a list of down sampling blocks
        for i in range(1, depth):
            self.downs.append(DownSample(channels[i-1], channels[i], p_drop))

        # Bottleneck convolution block
        self.bottleneck = ConvBlock(channels[-1], channels[-1], p_drop)

        self.ups = nn.ModuleList()
        dec_channels = list(reversed(channels))
        self.dec_blocks = nn.ModuleList()
        # Produce a list of up sampling blocks
        for i in range(depth - 1):
            up_output = dec_channels[i + 1]
            self.ups.append(nn.ConvTranspose2d(dec_channels[i], up_output, kernel_size=2, stride=2))
            input_block = up_output * 2
            self.dec_blocks.append(ConvBlock(input_block, up_output, p_drop))

        self.head = nn.Conv2d(dec_channels[-1], out_channels, kernel_size=1)
        self.aux_heads = None

        # Initialise blocks via kaiming initialisaion
        kaiming_initialization(self)

    def forward(self, x):
        skips = []
        x0 = self.encoder0(x)
        skips.append(x0)
        x = x0
        # Apply down sampling
        for d in self.downs:
            x = d(x)
            skips.append(x)

        x = self.bottleneck(x)

        # Apply up sampling
        for i in range(len(self.ups)):
            up = self.ups[i](x)
            skip = skips[-(i+2)]
            if up.shape[-2:] != skip[-2:]:
                up = Functional.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, up], dim=1)
            x = self.dec_blocks[i](x)

        out = self.head(x)
        return out


def kaiming_initialization(module: nn.Module):
    """
    Initialise network weights using Kaiming/He initialisation
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.01)  # LeakyReLU a=0.01
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class DiceLoss(nn.Module):
    """
    DICE loss measuring difference between prediction and truth

    Args:
        smoothing: Smoothing factor
        ignore_idx: Class index to ignore (for certain datasets)
    """
    def __init__(self, smoothing: float=1.0, ignore_idx: int=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_idx = ignore_idx

    def forward(self, predictor: torch.Tensor, target: torch.Tensor):
        # Convert to probabilities
        if predictor.shape[1] > 1:
            predictor = Functional.softmax(predictor, dim=1)
        else:
            predictor = torch.sigmoid(predictor)

        num_classes = predictor.shape[1]
        
        # Convert to one-hot encoding
        target_one_hot = Functional.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        if self.ignore_idx >= 0:
            mask = (target != self.ignore_idx).float().unsqueeze(1)
            predictor = predictor * mask
            target_one_hot = target_one_hot * mask

        # Calculate intersection and union between predictor and target
        intersection = (predictor * target_one_hot).sum(dim=(2, 3))
        union = predictor.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        # Calculate DICE coefficient per class
        dice = (2.0 * intersection + self.smoothing) / (union + self.smoothing)

        dice_loss = 1.0 - dice.mean()

        return dice_loss


def dice_coefficient(predictor: torch.Tensor, target: torch.Tensor, smoothing: float = 1e-6, threshold: float = 0.5):
    """
    Calculate DICE similarity coefficient for evaluation

    Args:
        predictor: Predictions
        target: Truth
        smoothing: Smoothing factor
        threshold: Threshold for binary predictions
    """
    if predictor.shape[1] > 1:
        predictor = torch.softmax(predictor, dim=1)
        predictor = torch.argmax(predictor, dim=1)  # (B, H, W)
    else:
        predictor = (torch.sigmoid(predictor) > threshold).float().squeeze(1)

    num_classes = max(target.max().item(), predictor.max().item()) + 1

    # Convert to one-hot encoding
    predictor_one_hot = Functional.one_hot(predictor.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = Functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Calculate intersection and union between prediction and truth
    intersection = (predictor_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = predictor_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

    # Calculate DICE similarity
    dice = (2.0 * intersection + smoothing) / (union + smoothing)

    return dice


if __name__ == "__main__":
    print("Testing Improved 2D UNet architecture...")

    model = IUNet2D(in_channels=1, out_channels=3, base_channel=32, depth=4)

    x = torch.randn(2, 1, 256, 256)

    output = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting Deep Supervision...")
    model_ds = IUNet2D(in_channels=1, out_channels=2, base_channel=32, depth=4)
    outputs = model_ds(x)
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")

    print("\nTesting Loss...")
    target = torch.randint(0, 2, (2, 256, 256))

    dice_loss = DiceLoss()

    loss_dice = dice_loss(output, target)

    print(f"Dice Loss: {loss_dice.item():.4f}")

    print("\nTesting metrics...")
    dice_scores = dice_coefficient(output, target)

    print(f"Dice Coefficients per Class: {dice_scores}")

    print("\nAll Tests Completed.")