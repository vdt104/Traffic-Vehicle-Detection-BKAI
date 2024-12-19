import argparse
from ultralytics import YOLO

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    parse.add_argument('--split', type=str, default='val', help='the split of data')
    parse.add_argument('--iou', type=float, default=0.5, help='the iou threshold')
    parse.add_argument('--conf', type=float, default=None, help='the confidence threshold')
    parse.add_argument('--name', type=str, default=None, help='the dir to save the model')
    parse.add_argument('--agnostic_nms', type=bool, default=False, help='use agnostic nms')

    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)

    model.val(
        data=args.data_dir, 
        split=args.split,
        iou=args.iou,
        conf=args.conf,
        save_txt=True,
        save_conf=True,
        name=args.name,
        agnostic_nms=args.agnostic_nms,
        project='runs/test',
    ) 

if __name__ == '__main__':
    main()
