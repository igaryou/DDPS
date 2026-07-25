from mmcv.runner.checkpoint import load_checkpoint

from mmseg.models.backbones import MixVisionTransformer
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

from ..utils.pretrained_checkpoint import load_validated_submodule


@BACKBONES.register_module()
class MixVisionTransformerCustomInitWeights(MixVisionTransformer):
    def __init__(self, strict_pretrained=False, **kwargs):
        self.strict_pretrained = strict_pretrained
        self.pretrained_load_report = None
        super().__init__(**kwargs)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type', None) == 'Pretrained'):
            pretrained = self.init_cfg['checkpoint']
            if isinstance(pretrained, str):
                logger = get_root_logger()
                if self.strict_pretrained:
                    load_validated_submodule(
                        self,
                        pretrained,
                        checkpoint_prefix='backbone',
                        description='MiT backbone',
                        required_target_prefixes=None,
                        fail_on_unexpected=True)
                else:
                    load_checkpoint(
                        self,
                        pretrained,
                        strict=False,
                        logger=logger,
                        revise_keys=[
                            (r'^module\.', ''), (r'^backbone\.', '')
                        ])
