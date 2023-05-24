# ==================== download checkpoint ====================
#!pip install gdown
import os
#os.chdir('./HairMapper')

# ==================== download checkpoint ====================
#!pip install gdown
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

print('Please wait util all models are downloaded...')


# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def download_from_google_drive(file_id,file_dst):
  downloaded = drive.CreateFile({'id':file_id})
  downloaded.FetchMetadata(fetch_all=True)
  downloaded.GetContentFile(file_dst)

checkpoints ={
    'StyleGAN2-ada-Generator.pth':
    {
        'url':'1EsGehuEdY4z4t21o2LgW2dSsyN3rxYLJ',
        'dir': './ckpts'
    },
    'e4e_ffhq_encode.pt':
    {
        'url':'1cUv_reLE6k3604or78EranS7XzuVMWeO',
        'dir': './ckpts'
    },
    'model_ir_se50.pth':
    {
        'url':'1GIMopzrt2GE_4PG-_YxmVqTQEiaqu5L6',
        'dir': './ckpts'
    },
    'face_parsing.pth':
    {
        'url':'1IMsrkXA9NuCEy1ij8c8o6wCrAxkmjNPZ',
        'dir': './ckpts'
    },
    'vgg16.pth':
    {
        'url':'1EPhkEP_1O7ZVk66aBeKoFqf3xiM4BHH8',
        'dir': './ckpts'
    }
}
for ckpt in checkpoints:
  name = ckpt
  url = checkpoints[name]['url']
  output_dir = checkpoints[name]['dir']
  os.makedirs(output_dir,exist_ok=True)
  output_path = os.path.join(output_dir,name)
  #gdown.download(url=url,output=output_path,quiet=False) # bug
  download_from_google_drive(file_id=url, file_dst=output_path)

classification_ckpt =[
        {'url':'1SSw6vd-25OGnLAE0kuA-_VHabxlsdLXL',
        'dir': './classifier/gender_classification'},
        {'url':'1n14ckDcgiy7eu-e9XZhqQYb5025PjSpV',
        'dir': './classifier/hair_classification'}
]
for clf_ckpt_dict in classification_ckpt:
  name = 'classification_model.pth'
  url = clf_ckpt_dict['url']
  output_dir = clf_ckpt_dict['dir']
  os.makedirs(output_dir,exist_ok=True)
  output_path = os.path.join(output_dir,name)
  #gdown.download(url=url,output=output_path,quiet=False)
  download_from_google_drive(file_id=url, file_dst=output_path)