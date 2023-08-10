import sys
sys.path.append(
    './mixformer_convmae/'
)
import datetime
import json
import torch
# for debug
import cv2
import os
from mixformer_convmae import (
        cfg,
        Preprocessor_wo_mask,
        TrackerParams,
        build_mixformer_convmae_online_score,
        clip_box,
        sample_target ,
        update_config_from_file
    )
from util import (Detection, Rect) 

class MixFormerOnline():
    def __init__(self, params):
        self.params = params
        network = build_mixformer_convmae_online_score(params.cfg,  train=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.frame_id = 0
        self.save_all_boxes = params.save_all_boxes
        
        self.update_interval = self.cfg.TEST.UPDATE_INTERVALS
        self.online_size = self.cfg.TEST.ONLINE_SIZES
        self.max_score_decay = 1.0
        # if not hasattr(params, 'vis_attn'):
        self.params.vis_attn = 0
        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Max score decay is ", self.max_score_decay)

    
    def initialize(self, image, info: dict):
        """
        init tracker baseline template  

        Args:
            image (np.ndarrat): initial frame from which template is to be taken
            info (dict): initial template bounding boxinit_info['init_bbox']

        Returns:
            None
        """        
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_state = info['init_bbox']
        
        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    @torch.no_grad()
    def update(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        if self.online_size==1:
            out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
        else:
            out_dict, _ = self.network.forward_test(search, run_score_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template)

            self.max_pred_score = -1
            self.online_max_template = self.template
        detection = Detection(
            Rect(*self.state),
            0,
            'target',
            pred_score
        )
        return {
            "target_bbox": self.state,
            "detection_object":[detection],
            "score":pred_score,
            "tempelate":self.online_template
            }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]



def build_mixformer_convmae_online(config_file = 'model_zoo/mae_online/baseline.yaml'):
    params = TrackerParams()
    update_config_from_file(config_file)
    params.cfg = cfg
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    print("search_area_scale: {}".format(params.search_factor))
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.save_all_boxes = False
    params.checkpoint = cfg.checkpoint
    model = MixFormerOnline(params)
    return model, params
    
    
def main():
    output = {
                'target_bbox': [],
                'time': [],
                'score':[],
                'tempelate':[]
                
            }
    def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)
    model,_ = build_mixformer_convmae_online()

    import time
    vid = cv2.VideoCapture('data/ftb-1.mp4')
    def read_frame():
        ret, frame = vid.read()    
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:pass
        finally:
            return frame, ret
        
    def skip_n(n=10):
        for i in range(n):
            read_frame()
    # skip_n(70)
    frame, ret = read_frame()
    
    if ret:
        frame_disp = frame.copy()
        cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (0, 0, 0), 1)
        x, y, w, h = cv2.selectROI('display_name', cv2.cvtColor(frame_disp, cv2.COLOR_RGB2BGR), fromCenter=False)
        cv2.destroyAllWindows()
        init_state = [x, y, w, h]
        init_info = {'init_bbox':init_state}
        model.initialize(frame, init_info)
        w,h= frame.shape[:2]
        frame_size = (h,w)
        name_id = datetime.datetime.now().__str__()
        writer= cv2.VideoWriter(
            f'data/track/{name_id}.mp4',
            cv2.VideoWriter_fourcc(*'MP4V'),
            30, 
            frame_size
        )
        while ret:
            start_time = time.time()
            # {
            # "target_bbox": self.state,
            # "score":pred_score,
            # "tempelate":self.online_template.copy()
            # }
            preds = model.update(frame)
            _store_outputs({
                "target_bbox": preds['target_bbox'],
                "score":preds['score'],
                "tempelate":preds['tempelate'].cpu().numpy().tolist(),
                'time': time.time()-start_time
                })
            r = list(map(int, preds['target_bbox']))
            if preds['score'] > 0.5:
                writer.write(
                    cv2.rectangle(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255,255,255),4)
                )
            # skip_n(5)
            frame, ret = read_frame()
        
        vid.release()
        writer.release()
        # with open(f'{name_id}.json', 'w')as f:
        #     json.dump(
        #         output, f, indent=0
        #     )
if __name__ == '__main__':
    main()