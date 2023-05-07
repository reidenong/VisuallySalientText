# Visually Salient Text (VST)
Visual saliency is the distinct quality which makes some items stand out from the others and grab our attention.

**This project uses Deep Learning to extract Salient text from an image using State-of-the-Art Visual Transformer Architecture.**

![image](https://user-images.githubusercontent.com/65756407/236673237-2a9748d0-d069-4493-8ed7-d2708eefb4ae.png)  


The model used is the [Visual Saliency Transformer](https://github.com/nnizhang/VST), which was trained on a synthetically generated dataset which focused on textual saliency considerations. This Dataset consists of images in the formats of news articles, memes, advertisements and other commonly found internet images. Usage of Text Saliency Models include filtering out noise in text-rich environments, as well as improving OCR quality when in the wild.  




## Examples of Text Saliency used with EasyOCR
![image](https://user-images.githubusercontent.com/65756407/236673752-ed6f8236-fd72-433f-956b-dcab1531ce52.png)
> Raw Text: TAROT PREDICTS HUNG HOUSE INDIA TODAY IN UTTAR PRADESH 540 INDIA EXCLUSIVE TODAY MAN WHO SPOKE T0 SAIFULLAH DURING ENCOUNTER JALu Msn B PM Iop SheeLA BaJaJ, TarOT CarD READER Mt indiatoday-in NeWS LUCKNOW ENCOUNTER FLASH Saifullah died in exchange 0f fire Pm


> Salient Text: TAROT PREDICTS HUNG HOUSE INDIA IN UTTAR PRADESH INDIA TODAY NeWS LUCKNOW ENCOUNTER FLASH Saifullah died in exchange of fire   


<br />

![image](https://user-images.githubusercontent.com/65756407/236676626-c9d1411c-ab89-479b-954e-415677aea8b7.png)
> Raw text: NEED TO LOSE 30 POUNDS? TRY SENSA  FREEI SENSA" is clinically proven to help you lose 30 Ibs without dieting or spending all your time working out: Just sprinkle on your food; eat and lose weight! GET A GYM BODY WIthout GOING TO THE GYM NO COUNTING CALORIES NO STIMULANTS NO PILLS for Doesn t taste of the your foodl Try SENSA'FREEI Mfll SensaOftercom /OKer (8001750-6971 VoI | CLINICALLY PROVEN: 100% SATISFACTION GUARANTEED: SENSA eocicgdodged Ca Cla npmnd nn nandtan GNCLVWcll Oeoi S S ne nite  Deantat #op hanehroloadceoh A A dtatd CCdedoDado nolcamnatndndniot enn GPECIa< OKI 6 change ncadans SENSA CL


> Salient Text: 30 POUNDS? TRY SENSA  FREEI GET A GYM BODY Try SENSA'FREEI 
<br />

## Usage of VST
### Directory Structure of Key Components
```
VisuallySalientText
├── VST_DEMO.ipynb
├── Models
    ├── PretrainedModels
    |   └── 80.7_T2T_ViT_t_14.pth.tar***
    ├── Checkpoints
    |   └── RGB_VST.pth***
    └── Decoder.py, Transformer.py, ...
├── Data
    ├── OCSD
    │   ├── OCSD-TR     (training set)
    │   │   ├── OCSD-TR-Image
    │   │   │   └── img0.jpg, img1.jpg, ...
    │   │   └── OCSD-TR-Mask
    │   │   │   └── img0.png, img1.png, ...
    │   │   └── OCSD-TR-Contour
    │   │   │   └── img0.png, img1.png, ...
    │   ├── OCSD-TE     (testing set)
    │   │   ├── images
    │   │   │   └── img0.jpg, img1.jpg...
...
```
(***) Create the directories and download their respective model/weights for ![PretrainedModels](https://drive.google.com/file/d/1OhMg6u3gEp959zClZD8pki280ksgg_-1/view?usp=share_link) and ![Checkpoints](https://drive.google.com/file/d/1-aFTAnS4yZoCwrr4X3j6JsEfALliVYcH/view?usp=share_link)
- The directory structure here is for the Optical Character Saliency Dataset, but will also work for any dataset with Image-Mask-Contour formatted directories 
- Due to the small, convoluted nature of optical characters, the Contour Masks are largely unecessary for text saliency and can be replaced with a copy of the saliency masks  

<br />

## Saliency Inference / Testing
For images in the directory *Data/Dataset/images/image0.jpg*


``` !python VST.py --test_paths Dataset/ ```  

<br />

## Saliency Mask Visualization (Overlay)
- Refer to SaliencyHeatmapVisualization.ipynb  

<br />

## Saliency-OCR Integration with EasyOCR
For images in the directory *Data/Dataset/images/image0.jpg* and masks in the directory *Predictions/Dataset/RGB_VST/*.


```$ python SalOCR.py --imagefilepath Data/Dataset/images/ --maskfilepath Predictions/Dataset/RGB_VST/ ```


Text Output will be in TextOutput/ in JSON format.  

<br />

## Training
```$ python VST.py --Training True --Testing False ```
