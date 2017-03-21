
# Default Fetch 500 images per classifier
def Fetch_Melanoma_Data(DataPath, width,height,depth):
    import os
    import random
    from PIL import Image
    import numpy as np

    LABEL0 = ['benign', 0]
    LABEL1 = ['malignant', 1]
       
    XSIZE = width
    YSIZE = height
    ZSIZE = depth

# Load in the images and create the relevant fields
BasePath = DataPath;
skin_data = list()

# Description of the Data
skin_data = {'DESCR': '', 'data': 0, 'images': 0, 'target': 0, 'target_names': 0 , 'image_names': ''}
skin_data['DESCR'] = "ISIC archived melanoma and benign images"

# Get Malignant image file paths
f = os.listdir(BasePath)

N = len(f)
f = random.sample(f, N)

# Declare Array to store data and images
skin_data['images'] = np.zeros((N, ZSIZE, YSIZE, XSIZE), dtype='uint8')
skin_data['target'] = np.zeros(N, dtype='uint64')
skin_data['target_names'] = []
skin_data['image_names']  = []
skin_data['target_names'].append(LABEL0[0])
skin_data['target_names'].append(LABEL1[0])
    
    # Loop through files, load, and store
    for i in range(N):
        print("%d of %d %s"%(i,N,f[i]))
        if f[i].startswith("."):
            continue
               
        skin_data['image_names'].append(f[i])
        
        # Load the image and convert to grayscale
        Img = np.array(Image.open(BasePath + f[i]).resize((XSIZE,YSIZE),Image.LANCZOS), np.float32)
        Img /= 255

        skin_data['images'][i, :, :, :] = Img.transpose((2, 0, 1))
      

        if( LABEL0[0] in f[i] ):
            skin_data['target'][i] = LABEL0[1]
        else:
            skin_data['target'][i] = LABEL1[1]
        
        #print("%s \t %s \t %d"%(f[i],skin_data['target_names'][skin_data['target'][i]],skin_data['target'][i]))

    return skin_data
