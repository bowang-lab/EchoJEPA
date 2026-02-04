# Standard library imports
import os
import math
import glob
import json
import pickle
import random

# Third-party library imports
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pydicom
import sklearn
import sklearn.metrics
import transformers


# Local module imports
import utils

class EchoPrime:
    def __init__(self, device=None, lang='en'):
        """
        Initialize EchoPrime
        
        Args:
            device: Device PyTorch cpu or cuda, it will automatically use cuda if available
            lang (str): language ('en' for english, 'it' for italian) - default to english
        """

        # load language specific files
        utils.initialize_language(lang)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("model_data/weights/echo_prime_encoder.pt",map_location=device)
        echo_encoder = torchvision.models.video.mvit_v2_s()
        echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
        echo_encoder.load_state_dict(checkpoint)
        echo_encoder.eval()
        echo_encoder.to(device)
        for param in echo_encoder.parameters():
            param.requires_grad = False
            
        vc_state_dict = torch.load("model_data/weights/view_classifier.pt")
        view_classifier = torchvision.models.convnext_base()
        view_classifier.classifier[-1] = torch.nn.Linear(
            view_classifier.classifier[-1].in_features, 11
        )
        view_classifier.load_state_dict(vc_state_dict)

        view_classifier.to(device)
        view_classifier.eval()
        for param in view_classifier.parameters():
            param.requires_grad = False
        
        self.echo_encoder = echo_encoder
        self.view_classifier = view_classifier
        self.frames_to_take=32
        self.frame_stride=2
        self.video_size=224
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        self.device=device
        self.lang = lang

        # load MIL weights per section
        self.MIL_weights = pd.read_csv("assets/MIL_weights.csv")
        self.non_empty_sections=self.MIL_weights['Section']
        self.section_weights=self.MIL_weights.iloc[:,1:].to_numpy()
    
        # Load candidate reports
        self.candidate_studies=list(pd.read_csv("model_data/candidates_data/candidate_studies.csv")['Study'])
        candidate_embeddings_p1=torch.load("model_data/candidates_data/candidate_embeddings_p1.pt")
        candidate_embeddings_p2=torch.load("model_data/candidates_data/candidate_embeddings_p2.pt")
        self.candidate_embeddings=torch.cat((candidate_embeddings_p1,candidate_embeddings_p2),dim=0)
        candidate_reports=pd.read_pickle("model_data/candidates_data/candidate_reports.pkl")
        self.candidate_reports = [utils.phrase_decode(vec_phr) for vec_phr in tqdm(candidate_reports)]
        self.candidate_labels = pd.read_pickle("model_data/candidates_data/candidate_labels.pkl")
        self.section_to_phenotypes = pd.read_pickle("assets/section_to_phenotypes.pkl")

    def process_dicoms(self,INPUT):
        """
        Reads DICOM video data from the specified folder and returns a tensor 
        formatted for input into the EchoPrime model.

        Args:
            INPUT (str): Path to the folder containing DICOM files.

        Returns:
            stack_of_videos (torch.Tensor): A float tensor of shape  (N, 3, 16, 224, 224)
                                            representing the video data where N is the number of videos,
                                            ready to be fed into EchoPrime.
        """

        dicom_paths = glob.glob(f'{INPUT}/**/*.dcm',recursive=True)
        stack_of_videos=[]
        for idx, dicom_path in tqdm(enumerate(dicom_paths),total=len(dicom_paths)):
            try:
                # simple dicom_processing
                dcm=pydicom.dcmread(dicom_path)
                pixels = dcm.pixel_array
                
                # exclude images like (600,800) or (600,800,3)
                if pixels.ndim < 3 or pixels.shape[2]==3:
                    continue 
                    
                # if single channel repeat to 3 channels    
                if pixels.ndim==3:
                    
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                # mask everything outside ultrasound region
                pixels=utils.mask_outside_ultrasound(dcm.pixel_array)
                
                
                
                #model specific preprocessing
                x = np.zeros((len(pixels),224,224,3))
                for i in range(len(x)):
                    x[i] = utils.crop_and_scale(pixels[i])
                
                x = torch.as_tensor(x, dtype=torch.float).permute([3,0,1,2])
                # normalize
                x.sub_(self.mean).div_(self.std)
            
                ## if not enough frames add padding
                if x.shape[1] < self.frames_to_take:
                    padding = torch.zeros(
                    (
                        3,
                        self.frames_to_take - x.shape[1],
                        self.video_size,
                        self.video_size,
                    ),
                    dtype=torch.float,
                    )
                    x = torch.cat((x, padding), dim=1)
                    
                start=0
                stack_of_videos.append(x[:, start : ( start + self.frames_to_take) : self.frame_stride, : , : ])
                
            except Exception as e:
                print("corrupt file")
                print(str(e))

        stack_of_videos=torch.stack(stack_of_videos)
        
        return stack_of_videos

    def process_mp4s(self,INPUT):
        """
        Reads MP4 video data from the specified folder and returns a tensor 
        formatted for input into the EchoPrime model.

        Args:
            INPUT (str): Path to the folder containing MP4 files.

        Returns:
            stack_of_videos (torch.Tensor): A float tensor of shape  (N, 3, 16, 224, 224)
                                            representing the video data where N is the number of videos,
                                            ready to be fed into EchoPrime.
        """

        dicom_paths = glob.glob(f'{INPUT}/**/*.mp4',recursive=True)
        stack_of_videos=[]
        for idx, dicom_path in enumerate(dicom_paths):
            try:
                # simple dicom_processing
                pixels,_,metadata = torchvision.io.read_video(dicom_path)
                fps=metadata['video_fps']
                pixels=np.array(pixels)

                #model specific preprocessing
                x = np.zeros((len(pixels),224,224,3))
                for i in range(len(x)):
                    x[i] = utils.crop_and_scale(pixels[i])

                x = torch.as_tensor(x, dtype=torch.float).permute([3,0,1,2])
                # normalize
                x.sub_(self.mean).div_(self.std)

                ## if not enough frames add padding
                if x.shape[1] < self.frames_to_take:
                    padding = torch.zeros(
                    (
                        3,
                        self.frames_to_take - x.shape[1],
                        self.video_size,
                        self.video_size,
                    ),
                    dtype=torch.float,
                    )
                    x = torch.cat((x, padding), dim=1)

                start=0
                stack_of_videos.append(x[:, start : ( start + self.frames_to_take) : self.frame_stride, : , : ])

            except Exception as e:
                print("corrupt file")
                print(str(e))

        stack_of_videos=torch.stack(stack_of_videos)

        return stack_of_videos

    def embed_videos(self,stack_of_videos):
        """
        Given a set of videos that belong to one echocardiogram study,
        embed them in the latent space using EchoPrime encoder
        
        Args:
            stack_of_videos (torch.Tensor): A float tensor of shape (N, 3, 16, 224, 224)
                                            with preprocessed echo video data
            
        Returns:
            stack_of_features (torch.Tensor) A float tensor of shape (N, 512)
                                            with latent embeddings corresponding to echo videos
        """
        bin_size=50
        n_bins=math.ceil(stack_of_videos.shape[0]/bin_size)
        stack_of_features_list=[]
        with torch.no_grad():
            for bin_idx in range(n_bins):
                start_idx = bin_idx * bin_size
                end_idx = min( (bin_idx + 1) * bin_size, stack_of_videos.shape[0])
                bin_videos = stack_of_videos[start_idx:end_idx].to(self.device)
                bin_features = self.echo_encoder(bin_videos)
                stack_of_features_list.append(bin_features)
            stack_of_features=torch.cat(stack_of_features_list,dim=0)
        return stack_of_features

    def get_views(self, stack_of_videos, visualize=False, return_view_list=False):
        """
        Args:
            stack_of_videos (torch.Tensor): A float tensor with preprocessed echo video data
            
        Returns:
            stack_of_view_encodings (torch.Tensor) A float tensor of one hot embeddings with shape (N, 11)
                                                representing echocardiogram views
        """
        ## get views   
        stack_of_first_frames = stack_of_videos[:,:,0,:,:].to(self.device)
        with torch.no_grad():
            out_logits=self.view_classifier(stack_of_first_frames)
        out_views=torch.argmax(out_logits,dim=1)
        view_list = [utils.COARSE_VIEWS[v] for v in out_views]
        stack_of_view_encodings = torch.stack([torch.nn.functional.one_hot(out_views,11)]).squeeze().to(self.device)

        # visualize images and the assigned views
        if visualize:
            print("Preprocessed and normalized video inputs")
            rows, cols = (len(view_list) // 12 + (len(view_list) % 9 > 0)), 12
            fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
            axes = axes.flatten()
            for i in range(len(view_list)):
                display_image = (stack_of_first_frames[i].cpu().permute([1,2,0]) * 255).numpy()
                display_image = np.clip(display_image, 0, 255).astype('uint8')
                display_image = np.ascontiguousarray(display_image)
                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                cv2.putText(display_image, view_list[i].replace("_"," "), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                axes[i].imshow(display_image)
                axes[i].axis('off')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

        if return_view_list:
            return view_list
            
        return stack_of_view_encodings
    @torch.no_grad()
    def encode_study(self,stack_of_videos,visualize=False):
        """
        Produces an EchoPrime embedding of the echocardiography study

        Args:
            stack_of_videos (torch.Tensor): A float tensor of shape (N, 3, 16, 224, 224)
                                            with preprocessed echo video data           
        Returns:
            encoded_study (torch.Tensor): A float tensor of shape (N, 523)
        """
        stack_of_features=self.embed_videos(stack_of_videos)
        stack_of_view_encodings=self.get_views(stack_of_videos,visualize)
        encoded_study = torch.cat( (stack_of_features ,stack_of_view_encodings),dim=1)
        
        return encoded_study
    
    def translate_sections(self, report:str) -> str:

        translations = {}

        if self.lang=='it':
            translations = {
            "Left Ventricle": "Ventricolo Sinistro",
            "Resting Segmental Wall Motion Analysis": "Cinetica Segmentaria a Riposo",
            "Right Ventricle": "Ventricolo Destro",
            "Left Atrium": "Atrio Sinistro",
            "Right Atrium": "Atrio Destro",
            "Atrial Septum": "Setto Inter-Atriale",
            "Mitral Valve": "Valvola Mitrale",
            "Aortic Valve": "Valvola Aortica",
            "Tricuspid Valve": "Valvola Tricuspide",
            "Pulmonic Valve": "Valvola Polmonare",
            "Pericardium": "Pericardio",
            "Aorta": "Aorta",
            "IVC": "Vena Cava Inferiore",
            "Pulmonary Artery": "Arteria Polmonare",
            "Pulmonary Veins": "Vene Polmonari",
            "Postoperative Findings": "Esiti Post-Operatori",
            }
        elif self.lang=='bs':
            translations = {
                "Left Ventricle": "Lijeva komora",
                "Resting Segmental Wall Motion Analysis": "Analiza segmentalne pokretljivosti stijenke u mirovanju",
                "Right Ventricle": "Desna komora",
                "Left Atrium": "Lijeva pretkomora",
                "Right Atrium": "Desna pretkomora",
                "Atrial Septum": "Interatrijski septum",
                "Mitral Valve": "Mitralni zalisak",
                "Aortic Valve": "Aortni zalisak",
                "Tricuspid Valve": "Trikuspidalni zalisak",
                "Pulmonic Valve": "Pulmonalni zalisak",
                "Pericardium": "Perikard",
                "Aorta": "Aorta",
                "IVC": "Donja šuplja vena",
                "Pulmonary Artery": "Plućna arterija",
                "Pulmonary Veins": "Plućne vene",
                "Postoperative Findings": "Postoperativni nalazi",
            }


        """
        elif self.lang=='your_language_code':
            translations = {
                # add your translations here
            }
        """

        for section, t in translations.items():
            report = report.replace(section, t)
        
        return report

    def generate_report(self, study_embedding: torch.Tensor) -> str:
        """
        Given the EchoPrime study embedding generate a report
        for each section focus on the views weighted
        Args:
            study_embedding - torch tensor of shape num_videos x 572
            original_report - text for original study
        """
        study_embedding=study_embedding.cpu()
        generated_report=""
        for s_dx, sec in enumerate(self.non_empty_sections):
            # need to multiply it based on what section does the view belong to.
            cur_weights=[self.section_weights[s_dx][torch.where(ten==1)[0]] for ten in study_embedding[:,512:]]
            no_view_study_embedding = study_embedding[:,:512] * torch.tensor(cur_weights,dtype=torch.float).unsqueeze(1)
            # weights by views.
            no_view_study_embedding=torch.mean(no_view_study_embedding,dim=0)
            no_view_study_embedding=torch.nn.functional.normalize(no_view_study_embedding,dim=0)
            similarities=no_view_study_embedding @ self.candidate_embeddings.T
            
            extracted_section="Section not found."
            while extracted_section=="Section not found.":
                max_id = torch.argmax(similarities)
                predicted_section = self.candidate_reports[max_id]
                extracted_section = utils.extract_section(predicted_section,sec)
                if extracted_section != "Section not found.":
                    generated_report+= extracted_section
                similarities[max_id]=float('-inf')

        if self.lang!='en':
            generated_report = self.translate_sections(generated_report)
                
        return generated_report
    
    def predict_metrics(self,study_embedding: torch.Tensor,
                    k=50) -> dict:
        """
        study_embedding is a set of embeddings of all videos from the study e.g (52,512)
        Takes a study embedding as input and
        outputs a dictionary for a set of 26 features
        """
        #per_section_study_embedding has shape (15,512)
        per_section_study_embedding=torch.zeros(len(self.non_empty_sections),512)
        study_embedding=study_embedding.cpu()
        # make per section study embedding
        for s_dx, sec in enumerate(self.non_empty_sections):
            # get section weights
            this_section_weights=[self.section_weights[s_dx][torch.where(view_encoding==1)[0]]
                        for view_encoding in study_embedding[:,512:]]
            this_section_study_embedding = (study_embedding[:,:512] * \
                                            torch.tensor(this_section_weights,
                                                        dtype=torch.float).unsqueeze(1))
            
            #weighted average
            this_section_study_embedding=torch.sum(this_section_study_embedding,dim=0)
            per_section_study_embedding[s_dx]=this_section_study_embedding
            
        per_section_study_embedding=torch.nn.functional.normalize(per_section_study_embedding)
        #similarities has shape (15,230676)
        similarities=per_section_study_embedding @ self.candidate_embeddings.T

        # for each row find indices of 50 highest values
        #top_candidate_ids has shape (15,50)
        top_candidate_ids=torch.topk(similarities, k=k, dim=1).indices
        #now predict for each phenotype:
        preds={}
        for s_dx, section in enumerate(self.section_to_phenotypes.keys()):
            for pheno in self.section_to_phenotypes[section]:
                preds[pheno] = np.nanmean([self.candidate_labels[pheno][self.candidate_studies[c_ids]]
                                    for c_ids in top_candidate_ids[s_dx]
                                        if self.candidate_studies[c_ids] in self.candidate_labels[pheno]])
        
        return preds

class EchoPrimeTextEncoder(torch.nn.Module):
    def __init__(self,device="cuda"):
        super().__init__()
        self.device=device
        config = transformers.AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        self.backbone = transformers.AutoModelForMaskedLM.from_config(config)
        self.text_projection = torch.nn.Linear(768, 512)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
        self.tokenizer.max_length=512
        self.to(device)
    def forward(self,report):
        text = self.tokenizer(
        report,
        padding="max_length",  # Pad to max_length
        max_length=512,        # Set the maximum length to 512 tokens
        truncation=True,        # Truncate if the input is longer than max_length,
        return_tensors="pt"
        )
        if text["input_ids"].shape[1] > 512:
            # find sep token positions
            sep_positions = list(
                torch.where(text["input_ids"].squeeze(0) == 3)[0].numpy()
            )

            # get maximum possible start that's not going to run out of tokens
            max_start = sep_positions[-1] - 512
            possible_starts = [pos for pos in sep_positions if pos < max_start]
            # add 0 as a possible start
            possible_starts.insert(0, 0)

            start = possible_starts[random.randint(0, len(possible_starts) - 1)]

            max_end = start + 512
            # find the first number less than max_end in sep_position
            for p in reversed(sep_positions):
                if p <= max_end:
                    end = p
                    break
            # finally cut the tokens
            text = transformers.BatchEncoding(
                data={k: v[:, start:end] for (k, v) in text.items()}
            )
        with torch.no_grad():
            text.to(self.device)
            text_emb = self.text_projection(
                self.backbone(**text, output_hidden_states=True).hidden_states[-1][
                    :, 0, :
                ]
            )
        return text_emb
