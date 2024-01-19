from timm.models import create_model
from models import models_eva

model = create_model(
    args.model,
    pretrained=False,
    img_size=224,
    num_classes=args.nb_classes,
    drop_rate=args.vit_dropout_rate,
    drop_path_rate=args.drop_path,
    attn_drop_rate=args.attn_drop_rate,
    drop_block_rate=None,
    use_mean_pooling=args.use_mean_pooling,
    use_checkpoint=args.use_checkpoint, 
)