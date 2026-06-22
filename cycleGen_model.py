import os
import torch
import torch.nn as nn

# ============================================================
# 1. CycleGAN용 Residual Block 정의
# ============================================================
class CycleGANResidualBlock(nn.Module):
    def __init__(self, features):
        super(CycleGANResidualBlock, self).__init__()
        # 체크포인트 구조에 맞춰 bias=True로 설정
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)

# ============================================================
# 2. CycleGAN Generator 아키텍처 (체크포인트 완전 호환 버전)
# ============================================================
class CycleGANGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()

        # [0~3] 초기 컨볼루션 블록
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # [4~9] Downsampling (특징 맵 축소)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # [10~18] 핵심 Residual Blocks
        for _ in range(num_residual_blocks):
            model += [CycleGANResidualBlock(in_features)]

        # [19~26] Upsampling (격자무늬 깨짐 방지를 위한 Upsample + Conv2d 조합)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # [27~29] 최종 출력 레이어 (28번 인덱스 Conv2d 매핑 완료)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0, bias=True),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ============================================================
# 3. 원터치 가중치 로드 및 추론 초기화 헬퍼 함수
# ============================================================
def load_cyclegan_model(checkpoint_path, device='cuda'):
    model = CycleGANGenerator(input_nc=3, output_nc=3, num_residual_blocks=9)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 키 파싱
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'netG_A2B' in checkpoint:
                state_dict = checkpoint['netG_A2B']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ [CycleGen] 가중치 로드 완료: {os.path.basename(checkpoint_path)}")
    else:
        print(f"⚠️ [CycleGen] 경고: 가중치 파일이 존재하지 않습니다. 경로: {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model
