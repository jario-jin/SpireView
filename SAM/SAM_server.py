#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
import threading
import time
import struct
import logging
from typing import List
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import onnxruntime
from pycocotools import mask


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
onnx_model_path = "sam_vit_h_4b8939.onnx"
device = "cuda"

ort_session = onnxruntime.InferenceSession(onnx_model_path)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


class SAMPipeline(threading.Thread):
    def __init__(self, client_key, client_socket, _server):
        threading.Thread.__init__(self)
        self.client_key = client_key
        self.client_socket = client_socket
        self._server = _server
        self.running = True
        self.img_h = 0
        self.img_w = 0
        self.img_id = 0
        self.img_embedding = None

    def run(self):
        while self.running:
            try:
                data = self.client_socket.recv(1024 * 1024)  # 1Mb
                # print("  data: {}".format(data[:4]))
                if data[: 2] == b'\xFA\xFC' and len(data) >= 12:
                    req_id = struct.unpack('I', data[4: 8])[0]
                    n_payloads = struct.unpack('I', data[8: 12])[0]
                    while len(data) < 12 + n_payloads:
                        data_ex = self.client_socket.recv(1024 * 1024)  # 1Mb
                        if data_ex == b"":
                            break
                        data += data_ex

                    return_msg = b'\xFB\xFD'

                    if data[2] == 0x00 and len(data) >= 12 + 8:  # 传Time
                        time_ms = struct.unpack('d', data[12: 20])[0]
                        print("time in ms = {}, id = {}".format(time_ms, req_id))
                        return_msg += b'\x00\xFF' + data[4: 8]
                        return_msg += struct.pack('I', 8)
                        return_msg += struct.pack('d', time_ms)
                        self.client_socket.send(return_msg)

                    if data[2] == 0x01 and len(data) >= 12 + n_payloads:  # 传图像数据
                        image = np.asarray(bytearray(data[12: 12 + n_payloads]), dtype="uint8")
                        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.img_h = image.shape[0]
                        self.img_w = image.shape[1]
                        self.img_id = req_id
                        predictor.set_image(image)
                        self.img_embedding = predictor.get_image_embedding().cpu().numpy()
                        print("h: {}, w: {}, id: {}, embedding: {}".format(
                            self.img_h, self.img_w, self.img_id, self.img_embedding.shape))
                        # cv2.imshow('image', image)
                        # cv2.waitKey(10)
                        # cv2.imwrite('/home/amov/img.jpg', image)
                        return_msg += b'\x01\xFF' + data[4: 8]
                        return_msg += struct.pack('I', 0)
                        self.client_socket.send(return_msg)

                    if data[2] == 0x02 and len(data) >= 12 + 8:  # 传单点坐标
                        if self.img_h > 0 and self.img_w > 0:
                            pt_x = struct.unpack('f', data[12: 16])[0] * self.img_w
                            pt_y = struct.unpack('f', data[16: 20])[0] * self.img_h
                            # print("pt (x, y) = ({}, {})".format(pt_x, pt_y))
                            """ -------------------- (START) Example point input ---------------------- """
                            input_point = np.array([[round(pt_x), round(pt_y)]])
                            input_label = np.array([1])

                            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
                            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(
                                np.float32)
                            onnx_coord = predictor.transform.apply_coords(onnx_coord, (self.img_h, self.img_w)).astype(
                                np.float32)

                            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                            onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                            ort_inputs = {
                                "image_embeddings": self.img_embedding,
                                "point_coords": onnx_coord,
                                "point_labels": onnx_label,
                                "mask_input": onnx_mask_input,
                                "has_mask_input": onnx_has_mask_input,
                                "orig_im_size": np.array((self.img_h, self.img_w), dtype=np.float32)
                            }

                            masks, _, low_res_logits = ort_session.run(None, ort_inputs)
                            masks = masks > predictor.model.mask_threshold
                            masks = masks.squeeze()
                            # masks_show = np.zeros_like(masks, dtype=np.uint8)
                            # masks_show[masks] = 255
                            # masks_show = cv2.resize(masks_show, (1280, 720))
                            # cv2.imshow('masks', masks_show)
                            # cv2.waitKey(10)
                            # cv2.imwrite('/home/amov/mask.jpg', masks_show)
                            enc_mask = mask.encode(masks.copy(order='F'))
                            return_msg += b'\x02\xFF' + data[4: 8]
                            return_msg += struct.pack('I', len(enc_mask['counts']))
                            return_msg += enc_mask['counts']
                            self.client_socket.send(return_msg)
                            """ -------------------- (END) Example point input ------------------------ """

                    if data[2] == 0x03 and len(data) >= 12 + 8:  # 传矩形框+点序列
                        if self.img_h > 0 and self.img_w > 0:
                            n_bboxes = struct.unpack('I', data[12: 16])[0]
                            n_pts = struct.unpack('I', data[16: 20])[0]
                            print("0x03 n_bboxes: {}, n_pts: {}".format(n_bboxes, n_pts))
                            if len(data) < 12 + 8 + n_bboxes * 16 + n_pts * 9:
                                continue
                            """ ----------------- (START) Example boxes&points input ------------------- """
                            input_box = np.array([])
                            if n_bboxes > 0:
                                input_box = np.array([
                                    round(struct.unpack('f', data[20: 24])[0] * self.img_w),
                                    round(struct.unpack('f', data[24: 28])[0] * self.img_h),
                                    round(struct.unpack('f', data[28: 32])[0] * self.img_w),
                                    round(struct.unpack('f', data[32: 36])[0] * self.img_h)
                                ])

                            input_point = []
                            input_label = []
                            for j in range(n_pts):
                                ind = 20 + n_bboxes * 16 + j * 9
                                input_point.extend([
                                    round(struct.unpack('f', data[ind: ind + 4])[0] * self.img_w),
                                    round(struct.unpack('f', data[ind + 4: ind + 8])[0] * self.img_h)
                                ])
                                print(input_point)
                                if data[ind + 8] == 0x00:
                                    input_label.append(0)
                                else:
                                    input_label.append(1)
                                print(input_label)

                            input_point = np.array(input_point).reshape(n_pts, 2)
                            input_label = np.array(input_label)

                            if len(input_box) == 4:
                                onnx_box_coords = input_box.reshape(2, 2)
                                onnx_box_labels = np.array([2, 3])
                            else:
                                onnx_box_coords = np.array([[0.0, 0.0]])
                                onnx_box_labels = np.array([-1])

                            onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
                            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(
                                np.float32)
                            onnx_coord = predictor.transform.apply_coords(onnx_coord, (self.img_h, self.img_w)).astype(
                                np.float32)
                            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                            onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                            ort_inputs = {
                                "image_embeddings": self.img_embedding,
                                "point_coords": onnx_coord,
                                "point_labels": onnx_label,
                                "mask_input": onnx_mask_input,
                                "has_mask_input": onnx_has_mask_input,
                                "orig_im_size": np.array((self.img_h, self.img_w), dtype=np.float32)
                            }

                            masks, _, _ = ort_session.run(None, ort_inputs)
                            masks = masks > predictor.model.mask_threshold
                            masks = masks.squeeze()
                            enc_mask = mask.encode(masks.copy(order='F'))
                            return_msg += b'\x03\xFF' + data[4: 8]
                            return_msg += struct.pack('I', len(enc_mask['counts']))
                            return_msg += enc_mask['counts']
                            self.client_socket.send(return_msg)
                            """ ----------------- (END) Example boxes&points input --------------------- """

                if data == b"":
                    self.running = False
                    self.client_socket.close()
                    self._server.quit(self.client_key)

            except Exception as e:
                print(e)
                self.running = False
                self.client_socket.close()
                self._server.quit(self.client_key)


class SAMTcpServer(threading.Thread):

    def __init__(self, port=9000):
        threading.Thread.__init__(self)
        socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_server.bind(('', port))
        socket_server.listen(5)
        print('Start listening on port {} ...'.format(port))
        self.socket_server = socket_server
        self.listening = True
        self.connected_clients = dict()

    def quit(self, client_key=None):
        if client_key is None:
            for k, c in self.connected_clients.items():
                c.close()
            self.listening = False
        else:
            del self.connected_clients[client_key]
        print("Now clients remain: {}".format(len(self.connected_clients)))

    def run(self):
        while self.listening:
            client_socket, client_address = self.socket_server.accept()
            client_key = '{}:{}'.format(client_address[0], client_address[1])
            print('Got client: [{}]'.format(client_key))

            self.connected_clients[client_key] = client_socket
            pipeline = SAMPipeline(client_key, client_socket, self)
            pipeline.start()


if __name__ == '__main__':
    server = SAMTcpServer(9093)
    server.start()
