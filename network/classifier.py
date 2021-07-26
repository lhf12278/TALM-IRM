import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 1.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class IdentityClassifier(nn.Module):

    def __init__(self, in_dim, source_pid_num):
        super(IdentityClassifier, self).__init__()

        self.in_dim = in_dim
        self.pid_num = source_pid_num

        self.classifier = nn.Linear(self.in_dim, self.pid_num, bias=False)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        cls_score = self.classifier(x.squeeze())
        return cls_score

class IdentityDomainClassifier(nn.Module):

    def __init__(self, in_dim, source_pid_num):
        super(IdentityDomainClassifier, self).__init__()

        self.in_dim = in_dim
        self.pid_num = source_pid_num + 1

        self.classifier = nn.Linear(self.in_dim, self.pid_num, bias=False)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        cls_score = self.classifier(x.squeeze())
        return cls_score

class CameraClassifier(nn.Module):
    def __init__(self, in_dim, source_cam, target_cam):
        super(CameraClassifier, self).__init__()
        self.in_dim = in_dim
        self.source_cam = source_cam
        self.target_cam = target_cam

        self.layer1 = self._make_layer(self.in_dim, 1024)
        self.layer2 = self._make_layer(1024, 512)
        self.layer3 = self._make_layer(512, 256)
        self.layer4 = self._make_layer(256, 128)
        self.layer5 = self._make_layer(128, 64)
        self.layer6 = self._make_layer(64, 32)
        self.fc = nn.Linear(32, self.source_cam + target_cam + 1)

    def _make_layer(self, in_nc, out_nc):
        block = [nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(out_nc),
                 nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*block)

    def forward(self, x):
        output = self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        output = self.fc(output.squeeze())
        return output