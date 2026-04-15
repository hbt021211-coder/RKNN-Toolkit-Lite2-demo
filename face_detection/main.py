import cv2
import argparse
import numpy as np
from rknnlite.api import RKNNLite
import time

class SCRFD():
    def __init__(self, model_path='./scrfd.rknn'):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.net = RKNNLite()
        self.net.load_rknn(model_path)
        self.net.init_runtime()
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw
    
    def distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    
    def detect(self, srcimg):
        # 1. 图像缩放
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = np.expand_dims(img, 0)
        
        # 2. 推理
        outs = self.net.inference(inputs=[img])
        
        # 核心逻辑：按照你提供的成功方案重新排列输出节点
        # outs[::3] 是所有层级的 score, outs[1::3] 是 bbox, outs[2::3] 是 kps
        outs = outs[::3] + outs[1::3] + outs[2::3]
        
        scores_list, bboxes_list, kpss_list = [], [], []
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # 获取当前层级的输出
            scores = outs[idx * self.fmc][0]
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            kps_preds = outs[idx * self.fmc + 2][0] * stride
            
            # --- 重要修改：高度和宽度必须根据当前的 inpHeight/Width 动态计算 ---
            height = self.inpHeight // stride
            width = self.inpWidth // stride
            
            # 生成锚点中心
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            # 阈值过滤
            pos_inds = np.where(scores >= self.confThreshold)[0]
            if len(pos_inds) == 0:
                continue
                
            # 计算坐标
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)
            
        if len(scores_list) == 0:
            return srcimg

        # 合并结果
        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        
        # 格式转换与原图映射
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        
        # NMS
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        
        # 绘制结果
        for i in indices:
            # 兼容 opencv 版本差异
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            
            xmin, ymin, w, h = bboxes[idx]
            xmax, ymax = int(xmin + w), int(ymin + h)
            xmin, ymin = int(xmin), int(ymin)
            
            cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
            for j in range(5):
                cv2.circle(srcimg, (int(kpss[idx, j, 0]), int(kpss[idx, j, 1])), 5, (0, 242, 255), thickness=-1)
            cv2.putText(srcimg, str(round(scores[idx], 3)), (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            
        return srcimg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./scrfd.rknn', help='model path')
    parser.add_argument('--cam_index', type=int, default=12, help='v4l2 device index')
    parser.add_argument('--ip', type=str, default='192.168.0.191', help='target ipv4 address')
    args = parser.parse_args()

    # 初始化模型
    mynet = SCRFD(args.model)

    # 1. 采集管线 (对应 C++ pipeline1)
    # 注意：Python中通常直接用设备索引，若要强制GStreamer，需写成字符串
    cap_pipeline = (f"v4l2src device=/dev/video{args.cam_index} io-mode=2 ! "
                    f"video/x-raw,format=NV12,width=640,height=640 ! "
                    f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2")
    
    cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)

    # 2. 发送管线 (对应 C++ send_pipeline)
    # mpph264enc 是 Rockchip 平台的硬编码器
    send_pipeline = (
        "appsrc is-live=true block=true format=time ! "
        "video/x-raw,format=BGR,width=640,height=640,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=NV12 ! "
        "mpph264enc ! h264parse ! rtph264pay pt=96 ! "
        "udpsink host=" + args.ip + " port=5000 sync=false async=false"
    )

    writer = cv2.VideoWriter(send_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 640), True)

    if not cap.isOpened() or not writer.isOpened():
        print("Error: Could not open Camera or VideoWriter")
        exit()

    print(f"Streaming to {args.ip}:5000 ... Press 'q' to quit (in terminal)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 图像处理与检测
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # 如果需要旋转，请确保尺寸匹配
            out_img = mynet.detect(frame)

            # 通过网络发送帧
            writer.write(out_img)

    except KeyboardInterrupt:
        pass

    cap.release()
    writer.release()
    print("Stream stopped.")