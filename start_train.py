import argparse
from ultralytics import YOLO

from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
from YoloWeightedDataset import YOLOWeightedDataset

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    # parse the arguments: yaml of the model, model, data_dir, epochs, imgsz
    parse.add_argument('--yaml', type=str, default=None, help='load the yaml of the model')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--data', type=str, default=None, help='the dir to data')
    parse.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parse.add_argument('--batch', type=int, default=16, help='batch size')
    parse.add_argument('--time', type=float, default=None, help='maximum training time in hours')
    parse.add_argument('--patience', type=int, default=100, help='number of epochs to wait without improvement before early stopping')
    parse.add_argument('--imgsz', type=int, default=640, help='the size of the image')
    parse.add_argument('--save', type=bool, default=True, help='enables saving of training checkpoints and final model weights')
    parse.add_argument('--save_period', type=int, default=-1, help='frequency of saving model checkpoints in epochs')
    parse.add_argument('--cache', type=str, default='False', help='enables caching of dataset images in memory or on disk')
    parse.add_argument('--device', type=str, default=None, help='computational device(s) for training')
    parse.add_argument('--workers', type=int, default=8, help='number of worker threads for data loading')
    parse.add_argument('--project', type=str, default=None, help='name of the project directory where training outputs are saved')
    parse.add_argument('--name', type=str, default=None, help='the dir to save the model')
    parse.add_argument('--exist_ok', type=bool, default=False, help='allows overwriting of an existing project/name directory')
    parse.add_argument('--pretrained', type=bool, default=True, help='start training from a pretrained model')
    parse.add_argument('--optimizer', type=str, default='auto', help='choice of optimizer')
    parse.add_argument('--verbose', type=bool, default=False, help='enables verbose output during training')
    parse.add_argument('--seed', type=int, default=0, help='sets the random seed for training')
    parse.add_argument('--deterministic', type=bool, default=True, help='forces deterministic algorithm use')
    parse.add_argument('--single_cls', type=bool, default=False, help='treats all classes as a single class')
    parse.add_argument('--rect', type=bool, default=False, help='enables rectangular training')
    parse.add_argument('--cos_lr', type=bool, default=False, help='utilizes a cosine learning rate scheduler')
    parse.add_argument('--close_mosaic', type=int, default=10, help='disables mosaic data augmentation in the last N epochs')
    parse.add_argument('--resume', type=bool, default=False, help='resumes training from the last saved checkpoint')
    parse.add_argument('--amp', type=bool, default=True, help='enables Automatic Mixed Precision (AMP) training')
    parse.add_argument('--fraction', type=float, default=1.0, help='specifies the fraction of the dataset to use for training')
    parse.add_argument('--profile', type=bool, default=False, help='enables profiling of ONNX and TensorRT speeds during training')
    parse.add_argument('--freeze', type=int, default=None, help='freezes the first N layers of the model')
    parse.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parse.add_argument('--lrf', type=float, default=0.01, help='final learning rate as a fraction of the initial rate')
    parse.add_argument('--momentum', type=float, default=0.937, help='momentum factor for SGD or beta1 for Adam optimizers')
    parse.add_argument('--weight_decay', type=float, default=0.0005, help='L2 regularization term')
    parse.add_argument('--warmup_epochs', type=float, default=3.0, help='number of epochs for learning rate warmup')
    parse.add_argument('--warmup_momentum', type=float, default=0.8, help='initial momentum for warmup phase')
    parse.add_argument('--warmup_bias_lr', type=float, default=0.1, help='learning rate for bias parameters during the warmup phase')
    parse.add_argument('--box', type=float, default=7.5, help='weight of the box loss component in the loss function')
    parse.add_argument('--cls', type=float, default=0.5, help='weight of the classification loss in the total loss function')
    parse.add_argument('--dfl', type=float, default=1.5, help='weight of the distribution focal loss')
    parse.add_argument('--pose', type=float, default=12.0, help='weight of the pose loss in models trained for pose estimation')
    parse.add_argument('--kobj', type=float, default=2.0, help='weight of the keypoint objectness loss in pose estimation models')
    parse.add_argument('--label_smoothing', type=float, default=0.0, help='applies label smoothing')
    parse.add_argument('--nbs', type=int, default=64, help='nominal batch size for normalization of loss')
    parse.add_argument('--overlap_mask', type=bool, default=True, help='determines whether segmentation masks should overlap during training')
    parse.add_argument('--mask_ratio', type=int, default=4, help='downsample ratio for segmentation masks')
    parse.add_argument('--dropout', type=float, default=0.0, help='dropout rate for regularization in classification tasks')
    parse.add_argument('--val', type=bool, default=True, help='enables validation during training')
    parse.add_argument('--plots', type=bool, default=False, help='generates and saves plots of training and validation metrics')
    parse.add_argument('--hsv_h', type=float, default=0.015, help='Adjusts the hue of the image by a fraction of the color wheel, introducing color variability.')
    parse.add_argument('--hsv_s', type=float, default=0.7, help='Alters the saturation of the image by a fraction, affecting the intensity of colors.')
    parse.add_argument('--hsv_v', type=float, default=0.4, help='Modifies the value (brightness) of the image by a fraction.')
    parse.add_argument('--degrees', type=float, default=0.0, help='Rotates the image randomly within the specified degree range.')
    parse.add_argument('--translate', type=float, default=0.1, help='Translates the image horizontally and vertically by a fraction of the image size.')
    parse.add_argument('--scale', type=float, default=0.5, help='Scales the image by a gain factor.')
    parse.add_argument('--shear', type=float, default=0.0, help='Shears the image by a specified degree.')
    parse.add_argument('--perspective', type=float, default=0.0, help='Applies a random perspective transformation to the image.')
    parse.add_argument('--flipud', type=float, default=0.0, help='Flips the image upside down with the specified probability.')
    parse.add_argument('--fliplr', type=float, default=0.5, help='Flips the image left to right with the specified probability.')
    parse.add_argument('--mosaic', type=float, default=1.0, help='Combines four training images into one.')
    parse.add_argument('--mixup', type=float, default=0.0, help='Blends two images and their labels, creating a composite image.')
    parse.add_argument('--agnostic_nms', type=bool, default=False, help='applies agnostic NMS to the final detections')
    parse.add_argument('--weight_dataset', type=bool, default=False, help='applies weighted dataset')
        
    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.yaml)
    model = YOLO(args.model)
    model = YOLO(args.yaml).load(args.model)

    # model = YOLO("yolov8l-pose.pt")

    # Apply weighted dataset for training with unbalanced classes
    if args.weight_dataset:
        build.YOLODataset = YOLOWeightedDataset

    model.train(
        data=args.data, 
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=args.name,
        batch=args.batch,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        pretrained=args.pretrained,
        verbose=args.verbose,
        seed=args.seed,
        deterministic=args.deterministic,
        single_cls=args.single_cls,
        rect=args.rect,
        cos_lr=args.cos_lr,
        resume=args.resume,
        amp=args.amp,
        val=args.val,
        save=args.save,
        save_period=args.save_period,
        cache=args.cache,
        # project=args.project,
        exist_ok=args.exist_ok,
        time=args.time,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        fraction=args.fraction,
        profile=args.profile,
        freeze=args.freeze,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        label_smoothing=args.label_smoothing,
        nbs=args.nbs,
        overlap_mask=args.overlap_mask,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        plots=args.plots,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        agnostic_nms=args.agnostic_nms,
        project="runs/train",
    )
    
    print("Train Loader Dataset: ", model.trainer.train_loader.dataset) 


if __name__ == '__main__':
    main()