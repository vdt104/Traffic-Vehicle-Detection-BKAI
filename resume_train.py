import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Resume YOLO model training')
    parser.add_argument('--model', type=str, required=True, help='Path to the partially trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        resume=True,
    )

if __name__ == '__main__':
    main()