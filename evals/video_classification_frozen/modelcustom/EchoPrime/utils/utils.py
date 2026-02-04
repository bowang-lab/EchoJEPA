import re
import json
import numpy as np
import functools 
import torchvision
from pathlib import Path
import numpy as np
import cv2
import pydicom as dicom
import torch

with open("assets/per_section.json", encoding="utf-8") as f:
    json_data = json.load(f)

_ybr_to_rgb_lut = None

COARSE_VIEWS=['A2C',
 'A3C',
 'A4C',
 'A5C',
 'Apical_Doppler',
 'Doppler_Parasternal_Long',
 'Doppler_Parasternal_Short',
 'Parasternal_Long',
 'Parasternal_Short',
 'SSN',
 'Subcostal']

ALL_SECTIONS=["Left Ventricle",
             "Resting Segmental Wall Motion Analysis",
             "Right Ventricle",
             "Left Atrium",
             "Right Atrium",
             "Atrial Septum",
             "Mitral Valve",
             "Aortic Valve",
             "Tricuspid Valve",
             "Pulmonic Valve",
             "Pericardium",
             "Aorta",
             "IVC",
             "Pulmonary Artery",
             "Pulmonary Veins",
             "Postoperative Findings"]


# global variables for language-specific data
phrases_per_section_list = None
phrases_per_section_list_org = None
regex_per_section = None


numerical_pattern = r'(\\d+(\\.\\d+)?)'  # Escaped backslashes for integers or floats
string_pattern = r'\\b\\w+.*?(?=\\.)'

def initialize_language(lang='en'):
    """
    Initialize language-specific variables.
    
    Args:
        lang (str): language code
    """
    all_phrases, t_list = None, None
    global phrases_per_section_list, phrases_per_section_list_org, regex_per_section
    
    if lang == 'en':
        phrases_file = "assets/all_phr.json"
    elif lang == 'it':
        phrases_file = "assets/all_phr_it.json"
    elif lang == 'bs':
        phrases_file = "assets/all_phr_bs.json"
    # add your translated file here
    #elif lang == 'your_language_code':
    #    phrases_file = "assets/all_phr_{your_language_code}.json"
    else:
        raise ValueError(f"Language not supported: {lang}, see README.md")
    
    # load the correct phrases file
    with open(phrases_file, encoding="utf-8") as f:
        all_phrases = json.load(f)
    
    t_list = {k: [all_phrases[k][j] for j in all_phrases[k]] 
              for k in all_phrases}
    
    # now we can create the other global variables
    phrases_per_section_list = {k: functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}
    phrases_per_section_list_org = {k: functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}

    regex_per_section = {k: make_it_regex(v)
                        for (k,v) in phrases_per_section_list.items()}
    
def isin(phrase,text):
    return phrase.lower() in (text.lower())

def extract_section(report, section_header):
    # Create a regex pattern that matches the section and anything up to the next [SEP]
    pattern = rf"{section_header}(.*?)(?=\[SEP\])"
    
    # Search for the pattern in the report
    match = re.search(pattern, report)
    
    # If a match is found, return the section including the header and the content up to [SEP]
    if match:
        # Include the trailing [SEP] if you need it as part of the output
        return f"{section_header}{match.group(1)}[SEP]"
    else:
        return "Section not found."

def extract_features(report: str) -> list:
    """
    Returns a list of 21 different features
    see json_data for a list of features
    """
    sorted_features=['impella',
    'ejection_fraction',
    'pacemaker',
    'rv_systolic_function_depressed',
    'right_ventricle_dilation',
    'left_atrium_dilation',
    'right_atrium_dilation',
    'mitraclip',
    'mitral_annular_calcification',
    'mitral_stenosis',
    'mitral_regurgitation',
    'tavr',
    'bicuspid_aov_morphology',
    'aortic_stenosis',
    'aortic_regurgitation',
    'tricuspid_stenosis',
    'tricuspid_valve_regurgitation',
    'pericardial_effusion',
    'aortic_root_dilation',
    'dilated_ivc',
    'pulmonary_artery_pressure_continuous']

    sorted_json_data = {k:json_data[k] for k in sorted_features}
    features=[]
    for key,value in sorted_json_data.items():
        if value['mode'] == "regression":
            match=None
            for phrase in value['label_sources']:
                pattern = re.compile((phrase.split("<#>")[0] + r"(\d{1,3}(?:\.\d{1,2})?)"), re.IGNORECASE)
                match = pattern.search(report)
                if match:
                    features.append(float(match.group(1)))
                    break
            if match is None:
                features.append(np.nan)

        elif value['mode'] == "binary":
            assigned=False
            for phrase in value['label_sources']:
                if isin(phrase,report):
                    features.append(1)
                    assigned=True
                    break
            if not assigned:
                features.append(0) 
    return features

def make_it_regex(sec):

    # replace numerical and string with corresponding regex
    for idx in range(len(sec)):
        sec[idx]=sec[idx].replace('(', '\(').replace(')', '\)').replace("+",'\+')
        sec[idx]=re.sub(r'<numerical>', numerical_pattern, sec[idx])
        sec[idx]=re.sub(r'<string>', string_pattern, sec[idx])

    regex_sec = re.compile('|'.join(sec), flags=re.IGNORECASE)
    return regex_sec


def remove_subsets(strings):
    result=[]
    for string in strings:
        if not any(string in res for res in result):
            result.append(string)  

    return list(result)

def structure_rep(rep):
    #remove double spaces
    rep = re.sub(r'\s{2,}', ' ', rep)
    structured_report = []
    for sec in ALL_SECTIONS:
        cur_section= extract_section(rep,sec)
        new_section=[sec+":"]
        
        # Find all matches using the combined pattern
        for match in re.finditer(regex_per_section[sec], cur_section):
            new_section.append(cur_section[match.start():match.end()])
            
        if len(new_section)>1:
            #remove phrases that are a subset of some other phrase
            new_section=remove_subsets(new_section)
            new_section.append("[SEP]")
            structured_report+=new_section
            
    # Join structured report parts
    structured_report = ' '.join(structured_report)
    return structured_report



def phrase_decode(phrase_ids):
    report = ""
    current_section = -1
    for sec_idx, phrase_idx, value in phrase_ids:
        section=list(phrases_per_section_list_org.keys())[sec_idx]
        if sec_idx!=current_section:
            if current_section!=-1:
                report+="[SEP] "
            report += section + ": "
            current_section=sec_idx

        # Get phrase template
        phr = phrases_per_section_list_org[section][phrase_idx]

        if '<numerical>' in phr:
            phr = phr.replace('<numerical>',str(value))
        elif '<string>' in phr:
            phr = phr.replace('<string>',str(value))
            
        report += phr + " "
    report += "[SEP]"
    return report


def apply_zoom(img_batch,zoom=0.1):
    """
    Apply zoom on a batch of images using PyTorch.
    
    Parameters:
        img_batch (torch.Tensor): A batch of images of shape (batch_size, height, width, channels).
        zoom (float): The zoom factor to apply, default is 0.1 (i.e., crop 10% from each side).
        
    Returns:
        torch.Tensor: A batch of zoomed images.
    """
    batch_size, height, width, channels = img_batch.shape

    # Calculate padding for zoom
    pad_x = round(int(width * zoom))  # X-axis (width)
    pad_y = round(int(height * zoom)) # Y-axis (height)

    # Crop the images by the zoom factor
    img_zoomed = img_batch[:, pad_y:-pad_y, pad_x:-pad_x, :]

    return img_zoomed

def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img

def downsample_and_crop(testarray):

        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0] # Start off the frameSum with the first frame<<
        # Convert color profile b/c cv2 messes up colors when it reads it in
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_BGR2GRAY)
        original = frame_sum
        frame_sum = np.where(frame_sum>0,1,0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.where(frame > 0, 1, 0) # make all non-zero values 1
            frame_sum = np.add(frame_sum, frame)
        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.dilate(np.uint8(frame_sum), kernel, iterations=10)
        # Make binary
        frame_overlap = np.where(frame_sum>0,1,0)                

        ###### Center and Square both Mask and Video ########        
        # Center image by finding center x of the image
        # Pick first 300 y-values
        center = frame_overlap[0:300, :]
        # compress along y axis
        center = np.mean(center, axis=0)
        try:
            center = np.where(center > 0, 1, 0) # make binary
        except:
            return
        # find index where first goes from 0 to 1 and goes from 1 to 0
        try:
            indexL = np.where(center>0)[0][0]
            indexR = center.shape[0]-np.where(np.flip(center)>0)[0][0]
            center_index = int((indexL + indexR) / 2)
        except:
            return
        # Cut off x on one side so that it's centered on x axis
        left_margin = center_index
        right_margin = center.shape[0] - center_index
        if left_margin > right_margin:
            frame_overlap = frame_overlap[:, (left_margin - right_margin):]
            testarray = testarray[:, :, (left_margin - right_margin):, :]
        else:
            frame_overlap = frame_overlap[: , :(center_index + left_margin)]
            testarray = testarray[:, :, :(center_index + left_margin), :]   

        #Make image square by cutting
        height = frame_overlap.shape[0]
        width = frame_overlap.shape[1]
        #Trim by 1 pixel if a dimension has an odd number of pixels
        if (height % 2) != 0:
            frame_overlap = frame_overlap[0:height - 1, :]
            testarray = testarray[:, 0:height - 1, :, :]
        if (width % 2) != 0:
            frame_overlap = frame_overlap[:, 0:width - 1]
            testarray = testarray[:, :, 0:width - 1, :]
        height = frame_overlap.shape[0]
        width = frame_overlap.shape[1]
        bias = int(abs(height - width) / 2)
        if height > width:
            frame_overlap = frame_overlap[bias:height-bias, :]
            testarray = testarray[:, bias:height-bias, :, :]
        else:
            frame_overlap = frame_overlap[:,bias:width-bias]
            testarray = testarray[:, :, bias:width-bias, :]
        return testarray

def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid

def write_video(p: Path, pixels: np.ndarray, fps=30.0, codec='h264'):
    torchvision.io.write_video(str(p), pixels, fps, codec)

def write_to_avi(frames: np.ndarray, out_file, fps=30):
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'MJPG'), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()

# def read_video(p: Path, start=None, end=None, units=None, out_format=None):
#     return torchvision.io.read_video(str(p), start, end, units, out_format)


def write_image(p: Path, pixels: np.ndarray):
    cv2.imwrite(str(p), pixels)


def ybr_to_rgb(pixels: np.array):
    lut = get_ybr_to_rgb_lut()
    return lut[pixels[..., 0], pixels[..., 1], pixels[..., 2]]


def get_ybr_to_rgb_lut(save_lut=True):
    global _ybr_to_rgb_lut

    # return lut if already exists
    if _ybr_to_rgb_lut is not None:
        return _ybr_to_rgb_lut
    
    # try loading from file
    lut_path = Path(__file__).parent / 'ybr_to_rgb_lut.npy'
    if lut_path.is_file():
        _ybr_to_rgb_lut = np.load(lut_path)
        return _ybr_to_rgb_lut

    # else generate lut
    a = np.arange(2 ** 8, dtype=np.uint8)
    ybr = np.concatenate(np.broadcast_arrays(a[:, None, None, None], a[None, :, None, None], a[None, None, :, None]), axis=-1)
    _ybr_to_rgb_lut = dicom.pixel_data_handlers.util.convert_color_space(ybr, 'YBR_FULL', 'RGB')
    if save_lut:
        np.save(lut_path, _ybr_to_rgb_lut)
    return _ybr_to_rgb_lut


def read_video(
    path,
    n_frames=None,
    sample_period=1,
    out_fps=None,
    fps=None,
    frame_interpolation=True,
    random_start=False,
    res=None,
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0,
    region=None  # (i_start, i_end, j_start, j_end)
):
    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    
    if res is not None:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)
    else:
        if region is None:
            out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
        else:
            out = np.zeros((n_frames, region[1] - region[0], region[3] - region[2]), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if region is not None:
            frame = frame[region[0]:region[1], region[2]:region[3]]
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps
