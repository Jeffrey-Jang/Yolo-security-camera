# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

import time
from datetime import datetime as dt
import datetime

import shutil

KERAS_PATH='./'
CAM_LOGFILE='./log/log_SecurityCam.log'
CAMPROC='./record/'
STOPCMD=CAMPROC + 'CAM_STOP'

# KERAS_PATH='/home/pi/keras-yolo3/'
# CAM_LOGFILE='/home/pi/skills/log/log_SecurityCam.log'
# CAMPROC='/home/pi/skills/record/'
# STOPCMD=CAMPROC + 'CAM_STOP'

# email
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

FROM_ADDRESS = 'Sending E-Mail'
MY_PASSWORD = 'Sending E-Mail PW'
TO_ADDRESS = 'Recieving E-Mail'
# BCC = 'raspi@mbox.re'
SUBJECT = '[Notice] Someome is in your room!'
BODY = '\nFrom RaspberryPi python3.'


# def create_message(from_addr, to_addr, bcc_addrs, subject, body, img_path):
def create_message(from_addr, to_addr, subject, body, img_path, file_name):
    msg = MIMEMultipart()
    msg.preamble = body
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    #msg['Bcc'] = bcc_addrs
    msg['Date'] = formatdate()
    with open(img_path, 'rb') as fp:
        img = MIMEImage(fp.read())
    img.add_header('Content-Disposition', 'attachment', filename=file_name)
    msg.attach(img)
    body = MIMEText(body.encode("utf-8"), body, 'utf-8')
    msg.attach(body)
    return msg


def send(from_addr, to_addrs, msg):
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login(FROM_ADDRESS, MY_PASSWORD)
    # smtpobj.sendmail(from_addr, to_addrs, msg.as_string())
    smtpobj.send_message(msg)
    smtpobj.close()

class YOLO(object):
    _defaults = {
#        "model_path": 'model_data/yolo.h5',
#        "anchors_path": 'model_data/yolo_anchors.txt',
        "model_path": KERAS_PATH + 'model_data/yolo-tiny.h5',
        "anchors_path": KERAS_PATH + 'model_data/tiny_yolo_anchors.txt',
        "classes_path": KERAS_PATH + 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        # Write Process number
        proc_path = CAMPROC + 'cam_process.txt'
        with open(proc_path, mode='w') as fproc:
            fproc.write(str(os.getpid()))
        fproc.close()
        #Init
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        
        with open(CAM_LOGFILE, mode='a') as flog:
            flog.write('{} model, anchors, and classes loaded.\n'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, log_path):
        InPerson = False
        #start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print(len(out_boxes))
        with open(log_path, mode='a') as f:
            f.write('{},'.format(len(out_boxes)))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            if predicted_class == 'person':
                InPerson = True
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            with open(log_path, mode='a') as f:
                f.write('{},{},{},{},{},{},'.format(predicted_class, score,left,top,right,bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        #end = timer()
        #print(end - start)
        with open(log_path, mode='a') as f:
            f.write('\n')
        return InPerson,image
        
    def close_session(self):
        self.sess.close()



def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    with open(CAM_LOGFILE, mode='a') as flog:
        flog.write('OUTPUT: {}\n'.format(output_path))
        flog.write('Process: {}\n'.format(str(os.getpid())))
    
    # Write detaction log 
    log_path = output_path + 'detaction_log.csv'
    with open(log_path, mode='w') as f:
        f.write('time,numbers,class,score,left,top,right,bottom,\n')
    
    isOutput = True if output_path != "" else False
    sended = False
    detacted = False
    captureOn = dt.now()
    while True:
        return_value, frame = vid.read()
        if isOutput:
            dsk = int(shutil.disk_usage(os.getcwd()).free / 1024 / 1024)
            if dsk < 100:
                with open(CAM_LOGFILE, mode='a') as flog:
                    flog.write("Disk remain: {:,d}[MB]\n".format(dsk))
                    flog.write('[ERROR] Not enough disk remain!!!\n')
                break
            tdatetime = dt.now()
            with open(log_path, mode='a') as f:
                f.write(tdatetime.strftime('%Y/%m/%d %H:%M:%S') + ',')
            video_name = tdatetime.strftime('%Y%m%d_%H%M%S') + '.png'
            if captureOn > tdatetime:
                cv2.imwrite(output_path + video_name,frame)

        image = Image.fromarray(frame)
        detacted,image = yolo.detect_image(image,log_path)
        result = np.asarray(image)
        ftext = "Recorded at: " + tdatetime.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(result, text=ftext, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #Displaying the result
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        if isOutput & detacted:
            #out.write(result)
            output_img = output_path + 'detacted/redult_' + video_name
            cv2.imwrite(output_img,result)
            if not sended:
                sended = True
                captureOn = dt.now()
                captureOn = captureOn + datetime.timedelta(minutes=30)
                msg = create_message(FROM_ADDRESS, TO_ADDRESS, SUBJECT, 
                                    tdatetime.strftime('%Y-%m-%d %H:%M:%S') + BODY, output_img, video_name)
                send(FROM_ADDRESS, TO_ADDRESS, msg)

        if captureOn < tdatetime:
            detacted = False
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        if os.path.isfile(STOPCMD):
            os.remove(STOPCMD)
            break
    yolo.close_session()
    f.close()
    
