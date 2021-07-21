# blazepose_pipeline__no_verbosity
This pipeline operates at 3 Hz with full model compared to full model mediapipeline in java operating 6 Hz on the same laptop. In this version of pipeline no results are being saved and input parameter is only --frames to be processed. 


**Installation: <br />**
Download trained weights from [Dropbox](https://www.dropbox.com/s/by94ilm6lzjrztd/extract_here_blazepose_inference.zip?dl=0) and unzip it using "extract here" command in the same directory  <br/>

**Report:  <br />**
Can be downloaded from here <br/>

**Dependencies: <br />**
Method 1: pip install tensorflow==2.0 pillow==8.0 jsonlib-python3 scipy==1.7.0 opencv-python==4.4.0.44 torch==1.9.0 <br/>

Method 2: 
- download this [file](https://www.dropbox.com/s/s7tyx1t4xuh8p7k/packages.zip?dl=0)
- unzip it at desired path
- change directory to the desired path
- run this command in cmd: <br>
for %x in (dir \*.*) do python -m pip install %x

<br>


Install Anaconda or python from the following [link](https://www.dropbox.com/s/yurh9gu4xz3lb0x/Anaconda3-2020.02-Windows-x86_64.exe?dl=0) respectively,<br>

**Simply run: <br />**
inference.py main_path = "folder with all files and weights" movie_path = "the path movie is saved" frames = #number of frames<br/>

**Help on every module used in this repository: **<br/>

Note: if you see a file apearing the the same name but ending in "_fn" then it means that it is reolicated as a function form <br/>

**aal_bp_post_proc_utils.py**<br/>
This file is the abbreviation of Altius Analytics Labs Blazepose Post Processing Utilities. Hence, it has all utilities being performed after running the blazepose including: <br/>

- Reversion of all coordinates from the cropped image to the original image<br/>
- post process hips and shoulder points<br/>
- saving results after alignment <br/>
- reversion of midpoints to the original image<br/>
- align images using midpoints<br/>
- smoothing/anti jittering coordinates after running blazepose<br/>

**aal_bp_utilities.py**<br/>
This file is the abbreviation of Altius Analytics Labs Blazepose Utilities. Hence, it has all utilities being performed preparing the data for blazepose including: <br/>

- delete previous results<br/>
- create folders<br/>
- aligning the image<br/>
- saving intermediate results<br/>
- performing tracking/blazepose based on the conditions <br/>
- image rotation using angular cropping<br/>
- functions required for angular croping, rotation of coordinates, rotating back the coordinates<br/>
- applying blazeface<br/>
- converting and storing blazeface coordinates for further processing of alignment<br/>
- showing midpoints results<br/>
- calculate body angle<br/>
- showing faces after blazeface<br/>
- auxiliary plotting of blazeface results provided by blazeface main repo<br/>

**bface_utils.py**<br/>
This file is the abbreviation of Blazeface Utilities. Hence, it has all utilities being performed for running the blazepose including: <br/>

- loading weights<br/>
- instatiating models using the blazeface class<br/>
- setting parameters such as min_score_thresh and min_suppression_threshold <br/>

**bpose_utils.py**<br/>
This file is the abbreviation of Blazepose Utilities. Hence, it has all utilities being performed for running the blazepose including: <br/>

- loading weights<br/>
- setting addresses<br/>
- instatiating models using the blazepose class<br/>

**blazepose_pipeline_inference.py** <br/>
This file and its function version "blazepose_pipeline_inference_fn.py" both tell the sequence happening for blazepose as a story<br/>

**inference.py**<br/>
This is the main file getting arguments from the user <br/>


 

