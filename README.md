# Human Pose Estimation (HPE) in basketball videos

## How to run

### Prerequisites
 - Linux or macOS
 - Python 3.6+
 - Pytorch
   - `pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
 - MMDetection
   - `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html`
   - `git clone https://github.com/open-mmlab/mmdetection.git`
   - `cd mmdetection`
   - `pip install -r requirements/build.txt`
   - `pip install -v -e .`
   - `cd ..`
 - MMPose  
   - `pip install -r requirements.txt`
   - `python setup.py develop`
   
### Run

`sh run.sh ${video_path} ${device}`

- `video_path` is path of basket ball video input
- `device`: `cpu` for using MMPose with CPU or `cuda` if using GPU
- the outputs is in new created `output` folder
