import pytorch_lightning as pl
from barcode_segmentation.models.model import BarcodeSegmentationModel
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog

# Load data and configuration
cfg = get_cfg()
# Регистрируем датасет (как в исходном коде)
data_dir = "/content/barcodes"  # Change this to your directory
DatasetCatalog.register("barcode_train", lambda d=data_dir: get_barcode_dicts(d))
MetadataCatalog.get("barcode_train").set(thing_classes=["barcode"])
barcode_metadata = MetadataCatalog.get("barcode_train")

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("barcode_train",)
cfg.DATASETS.TEST = () # No test dataset for now
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300   # adjust up if val losses are still going down, this is epochs
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (barcode)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Use Detectron2's data loader
train_loader = build_detection_train_loader(cfg, mapper=None)

# Instantiate the Lightning model
model = BarcodeSegmentationModel(cfg)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=10)  # Set the number of epochs
trainer.fit(model, train_loader)
