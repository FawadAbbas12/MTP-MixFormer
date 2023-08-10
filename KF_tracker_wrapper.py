from __future__ import annotations
import cv2
import datetime
import numpy as np
from mixformer_convmae_online import build_mixformer_convmae_online
from util import (Color, BaseAnnotator, TextAnnotator) 
from util import COLORS, THICKNESS
from util.KF import KalmanBoxTracker
from util.KF import associate, k_previous_obs
import time 
class TrackingModel_2D():
    def __init__(self) -> None:
        super().__init__()
        self.text_annotator = TextAnnotator(background_color=Color(255,255,255), text_color=Color(0,0,0), text_thickness=1) # draws text on detected image 
        self.annotator = BaseAnnotator( # draws eleptical on detected image
            colors=COLORS, 
            thickness=THICKNESS)
        
        self.output = {
                    'target_bbox': [],
                    'time': [],
                    'score':[],
                    'tempelate':[]
                }
        self.model, self.params = build_mixformer_convmae_online()
    def _store_outputs(self, tracker_out: dict, defaults=None):
                defaults = {} if defaults is None else defaults
                for key in self.output.keys():
                    val = tracker_out.get(key, defaults.get(key, None))
                    if key in tracker_out or val is not None:
                        self.output[key].append(val)
    
    def track(self, seq = 'data/ftb-1.mp4'):
        vid = cv2.VideoCapture(seq)
        def read_frame():
            ret, frame = vid.read()    
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:pass
            finally:return frame, ret
        frame, ret = read_frame()
        
        if ret:
            frame_disp = frame.copy()
            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (0, 0, 0), 1)
            init_state = list(cv2.selectROI('display_name', cv2.cvtColor(frame_disp, cv2.COLOR_RGB2BGR), fromCenter=False))
            cv2.destroyAllWindows()
            kf_init_bbox = [init_state[0],init_state[1],init_state[0]+init_state[2], init_state[1]+init_state[3]]
            init_info = {'init_bbox':init_state}
            kf = KalmanBoxTracker(kf_init_bbox, 0, 5)
            self.model.initialize(frame, init_info)
            w,h= frame.shape[:2]
            frame_size = (h,w)
            name_id = datetime.datetime.now().__str__().replace(':', '_')
            with open(f'data/track/KF_{name_id}.params','w') as f:
                init_info['params'] = self.params._to_dict()
                init_info['input_seq'] = seq
                import json
                json.dump(
                    init_info,
                    f,
                    indent=4   
                )
            writer= cv2.VideoWriter(
                f'data/track/KF_{name_id}.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                30, 
                frame_size
            )
            update = True
            last_time = time.time()
            while ret:   
                _kf_pred = kf.predict()[0]
                kf_pred = list(map(int,_kf_pred))
                preds = self.model.update(frame)
                r = preds['target_bbox']
                update_bbox = np.array(
                    [
                        r[0],r[1],r[0]+r[2], r[1]+r[3]
                    ]
                )
                # kf.update(update_bbox, 0)
                #def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):    
                matched, unmatched_dets, unmatched_trks,_ = associate(
                    np.array([update_bbox]),
                    np.ma.compress_rows(np.ma.masked_invalid([_kf_pred])), 
                    0.1,#Thresh IOU  
                    np.array([kf.velocity if kf.velocity is not None else np.array((0, 0)) ]), 
                    np.array([k_previous_obs(kf.observations, kf.age, kf.delta_t)]), 
                    0.2 # inertial
                    )
                # print(matched, unmatched_dets, unmatched_trks)
                if len(unmatched_dets):
                    if update:
                        last_time = time.time()
                        update = False
                        print('retaining KF output')
                    if time.time() - last_time > 3:
                        update = True
                        print('Updating based on Model output')
                else:update = True
                kf.update(update_bbox if update else _kf_pred, 0)
                t = preds['tempelate'].cpu().numpy().transpose(0,2,3,1)
                if t.shape[0] > 1:
                    t_0 = t[0]
                    for t_ in t[1:]:
                        t_0 = np.hstack((t_0, t_))
                    t = t_0
                else:
                    t =t[0] 
                mean = np.array([ 0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                t = std * t + mean
                t = np.clip(t, 0, 1)*255
                t_h,t_w,_=t.shape 
                im_h,im_w,_ = frame.shape
                frame[im_h-t_h:,im_w-t_w:,:] = t.astype(np.uint8)
                image = self.annotator.annotate(
                    image=frame, detections=preds['detection_object'])
                image = self.text_annotator.annotate(
                    image=image, detections=preds['detection_object'])
                writer.write(
                    cv2.rectangle(
                        cv2.cvtColor(image,cv2.COLOR_RGB2BGR),
                        (kf_pred[0], kf_pred[1]),
                        (kf_pred[2], kf_pred[3]),
                        (0,0,255),
                        4
                    )
                )
                frame, ret = read_frame()
            vid.release()
            writer.release()
 
 
if __name__ == '__main__':
    tracker = TrackingModel_2D()
    tracker.track('data/f1.mp4')