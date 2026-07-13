
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Project gating signal to intermediate dimensions
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Project encoder features to intermediate dimensions
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

      
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        
        g1 = self.W_g(g)      
        x1 = self.W_x(x)      

        psi = self.relu(g1 + x1)    
        psi = self.psi(psi)          

        return x * psi               


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class DecoderBlock(nn.Module):
   
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        bilinear: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Upsampling
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2,
            )

        # Attention gate (optional — for ablation)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(
                F_g=out_channels,     
                F_l=skip_channels,    
                F_int=out_channels // 2,  
            )

        # Double conv after concatenation
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout_rate)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        
        x = self.up(x)  

        
        if x.shape != skip.shape:
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])

        # Apply attention to encoder features
        if self.use_attention:
            skip = self.attention(g=x, x=skip)

        # Concatenate and process
        x = torch.cat([x, skip], dim=1)  # (B, out_ch + skip_ch, H, W)
        return self.conv(x)


#ATTENTION-UNET ARCHITECTURE IMPLEMENTATION
class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        base_features: int = 32,
        encoder_depth: int = 4,
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        bilinear: bool = False,
        encoder_name: Optional[str] = None,
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()
        self.depth = encoder_depth
        self.use_attention = use_attention
        self.encoder_name = encoder_name

        if encoder_name is not None:
            # --- Transfer Learning Path ---
            import segmentation_models_pytorch as smp
            self.encoder = smp.encoders.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
            enc_channels = self.encoder.out_channels
            
            self.decoders = nn.ModuleList()
            dec_out_channels = [base_features * (2 ** i) for i in reversed(range(encoder_depth))]
            
            in_ch = enc_channels[-1]
            for i in range(encoder_depth):
                skip_ch = enc_channels[-2 - i]
                out_ch = dec_out_channels[i]
                self.decoders.append(
                    DecoderBlock(
                        in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch,
                        use_attention=use_attention, bilinear=bilinear, dropout_rate=dropout_rate,
                    )
                )
                in_ch = out_ch
            self.output_conv = nn.Conv2d(dec_out_channels[-1], out_channels, kernel_size=1)
            
        else:
            # --- From Scratch Path ---
            channels = [base_features * (2 ** i) for i in range(encoder_depth + 1)]
            self.enc_first = ConvBlock(in_channels, channels[0], dropout_rate=0.0)
            
            self.encoders = nn.ModuleList()
            for i in range(encoder_depth):
                self.encoders.append(EncoderBlock(channels[i], channels[i + 1], dropout_rate))
                
            self.decoders = nn.ModuleList()
            for i in range(encoder_depth):
                dec_in = channels[encoder_depth - i]
                skip_ch = channels[encoder_depth - i - 1]
                dec_out = channels[encoder_depth - i - 1]
                self.decoders.append(
                    DecoderBlock(
                        in_channels=dec_in, skip_channels=skip_ch, out_channels=dec_out,
                        use_attention=use_attention, bilinear=bilinear, dropout_rate=dropout_rate,
                    )
                )
            self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_features = []
        if self.encoder_name is not None:
            features = self.encoder(x)
            skip_features = features[:-1]
            x = features[-1]
        else:
            x = self.enc_first(x)
            skip_features.append(x)
            for encoder in self.encoders:
                x = encoder(x)
                skip_features.append(x)
            x = skip_features.pop()

        for decoder in self.decoders:
            skip = skip_features.pop()
            x = decoder(x, skip)

        return self.output_conv(x)

    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        attention_maps = []
        skip_features = []
        if self.encoder_name is not None:
            features = self.encoder(x)
            skip_features = features[:-1]
            out = features[-1]
        else:
            out = self.enc_first(x)
            skip_features.append(out)
            for encoder in self.encoders:
                out = encoder(out)
                skip_features.append(out)
            out = skip_features.pop()

        for decoder in self.decoders:
            skip = skip_features.pop()
            up = decoder.up(out)
            if up.shape != skip.shape:
                diff_h = skip.size(2) - up.size(2)
                diff_w = skip.size(3) - up.size(3)
                up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
            
            if decoder.use_attention:
                g1 = decoder.attention.W_g(up)
                x1 = decoder.attention.W_x(skip)
                psi = decoder.attention.relu(g1 + x1)
                psi = decoder.attention.psi(psi)
                attention_maps.append(psi.detach())
                attended_skip = skip * psi
            else:
                attended_skip = skip
                attention_maps.append(None)
                
            cat = torch.cat([up, attended_skip], dim=1)
            out = decoder.conv(cat)

        return attention_maps

def build_model(config: Dict[str, Any]) -> AttentionUNet:
    mc = config["model"]
    model = AttentionUNet(
        in_channels=mc.get("in_channels", 3),
        out_channels=mc.get("out_channels", 4),
        base_features=mc.get("base_features", 32),
        encoder_depth=mc.get("encoder_depth", 4),
        dropout_rate=mc.get("dropout_rate", 0.2),
        use_attention=mc.get("use_attention", True),
        bilinear=mc.get("bilinear", False),
        encoder_name=mc.get("encoder_name", None),
        encoder_weights=mc.get("encoder_weights", "imagenet"),
    )
    return model
