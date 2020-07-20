import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz)
        # torch.save(model.state_dict(), "yolo_v4_mayank.pt")
        ONNX_EXPORT=True
        if ONNX_EXPORT:
            imgsz_tuple=(imgsz,imgsz)
            model.fuse()
            img = torch.zeros((1, 3) + imgsz_tuple)  # (1, 3, 320, 192)
            f = "custom_yolov4.onnx"  # *.onnx filename
            torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                              input_names=['images'], output_names=['classes', 'boxes'])

            # Validate exported model

        # Save model
        # save = (not opt.nosave) or (final_epoch and not opt.evolve)
        # if save:
        #     with open(results_file, 'r') as f:  # create checkpoint
        #         ckpt = {'epoch': epoch,
        #                 'best_fitness': best_fitness,
        #                 'training_results': f.read(),
        #                 'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
        #                 'optimizer': None if final_epoch else optimizer.state_dict()}
        #
        #     # Save last, best and delete
        #     torch.save(ckpt, last)
        #     if (best_fitness == fi) and not final_epoch:
        #         torch.save(ckpt, best)
        #     del ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='yolov4_modfied.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='../data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov4.weights', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
