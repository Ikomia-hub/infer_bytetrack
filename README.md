<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_bytetrack</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_bytetrack">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_bytetrack">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_bytetrack/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_bytetrack.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Multiple object tracking algorithm for object detection using ByteTrack.

![Example git](https://github.com/ifzhang/ByteTrack/blob/main/assets/MOT17-07-SDP.gif?raw=true)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add ByteTrack tracking algorithm
tracking = wf.add_task(name="infer_bytetrack", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    wf.run_on(array=frame)

    # Get results
    image_out = tracking.get_output(0)
    obj_detect_out = tracking.get_output(1)

    # Display
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(obj_detect_out), cv2.COLOR_BGR2RGB)
    display(img_res, title="ByteTrack", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **conf_thres** (float) - Default '0.25': Confidence threshold
- **conf_thres_match** (float) - Default '0.7': Threshold for considering an assignment valid.
- **track_buffer** (int) - Default '30': Buffer size.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_bytetrack", auto_connect=True)

algo.set_parameters({
    "param1": "value1",
    "param2": "value2",
    ...
})
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add algorithm
track = wf.add_task(name="infer_bytetrack", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_work.jpg")

# Iterate over outputs
for output in track.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

