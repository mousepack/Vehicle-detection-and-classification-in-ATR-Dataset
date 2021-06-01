import numpy as np
import keras
import random
from math import ceil
import cv2
from glob import glob 
import pickle as p

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_list, batch_size=16, input_dim=(5,224,224), output_dim=(224,224), n_channels=0, cropped =False,
                 offset=5,viz = False, folder='./atr/scaled2500_1to2/images/'):  
        'Initialization'
        self.folder = folder
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.data_list = data_list
        
        self.n_channels = n_channels
        self.viz = viz
        self.cropped = cropped
        self.offset = offset
        self.iteration =0
        # print('offset = %d, cropped_images: %s @ dim : %s'%(self.offset,str(self.cropped),str(self.input_dim)))
        self.on_epoch_end()

        self.indexes = [ [j for j in range(len(i))] for i in self.data_list]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list)*len(self.data_list[0]) / self.batch_size))
        # return(int(5))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Find list of IDs
        # import ipdb;ipdb.set_trace()
        list_IDs= []
        # for batch in range(self.batch_size):
        for idx,i in enumerate((self.indexes[0])):
            iteration = self.iteration*16+idx
            try:
                # self.indexes[0].remove(iteration)          
                list_IDs.append((0,iteration))
            except:
                # import ipdb;ipdb.set_trace()
                # return(False)
                continue
                # if self.iteration==
            if len(list_IDs) == 16:
                self.iteration+=1
                break
            # idx = random.randint(0,(len(self.indexes)-1))
            # while(self.indexes[idx] == []):
            #     idx = random.randint(0,(len(self.indexes)-1))
            # selected_index = random.choice(self.indexes[idx])
            # list_IDs.append((idx,selected_index))
            # self.indexes[idx].remove(selected_index)
        # Generate data
        X, y = self.__data_generation(list_IDs)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #reinitialize index
        self.indexes = [ [j for j in range(len(i))] for i in self.data_list] 

    def _fix_size(self,idx,selected_id):
        start_index = max(0,selected_id-2)
        end_index = min(selected_id+3,len(self.data_list[idx]))
        data_sample = self.data_list[idx][start_index:end_index]
        # import ipdb;ipdb.set_trace()
        while(len(data_sample) != 5):
            if end_index < 10:
                data_sample.insert(0,None)
            else:
                data_sample.append(None)

        return(data_sample)

    def _create_seg(self,data,image_size,size,offset):
        image_seg = np.zeros(size)
        for target in data['targets']:
            ul = np.array(target['ul'])-offset 
            br = np.array(target['ul'])+2*(np.array(target['center'])-np.array(target['ul']))+offset
            ul_re = np.array([(ul[1]/image_size[0])*size[0],(ul[0]/image_size[1])*size[1]]).astype(int)
            br_re = np.array([(br[1]/image_size[0])*size[0],(br[0]/image_size[1])*size[1]]).astype(int)
            image_seg[ul_re[0]:br_re[0],ul_re[1]:br_re[1]] = 255
        # import ipdb;ipdb.set_trace()
        return(image_seg,[(ul_re[1]+br_re[1])/2.0,(ul_re[0]+br_re[0])/2.0])
    
    # def _create_seg_circle(self,data,image_size,size,offset):
    #     seg = np.zeros(size)
    #     for target in data['targets']:
    #         center = np.array(target['center'])[::-1]

    #         center = center.astype(np.int16)
    #         ul = np.array(target['ul'])-offset 
    #         br = np.array(target['ul'])+2*(np.array(target['center'])-np.array(target['ul']))+offset

    #         ul_re = np.array([(ul[0]/image_size[0])*size[0],(ul[1]/image_size[1])*size[1]]).astype(int)
    #         br_re = np.array([(br[0]/image_size[0])*size[0],(br[1]/image_size[1])*size[1]]).astype(int)
    #         c= np.average(np.array([ul_re,br_re]),axis=0)
    #         cv2.circle(seg, tuple(c), radius=2,color = 255, thickness=4)

    #     # seg_resized = cv2.resize(seg,tuple(size[::-1]))
    #     return(image_seg,c)

    def print_dog(self,data):
        x = np.random.randint(100)
        for i in data:
            cv2.imwrite('x/dog-%d-%d.jpg'%(x,np.random.randint(10)),i)
        return(True)

    def norm(self,img):
        val_min = float(img.min())
        val_max = float(img.max())
        img_norm = (img-val_min)/(val_max-val_min)*255.0
        return(img_norm.astype(np.uint8))

    def get_DoG(self,data):
        diff_1 = self.norm(cv2.absdiff(data[0],data[2]))
        diff_2 = self.norm(cv2.absdiff(data[1],data[2]))
        diff_3 = self.norm(cv2.absdiff(data[3],data[2]))
        diff_4 = self.norm(cv2.absdiff(data[4],data[2]))
        # self.print_dog([diff_1,diff_2,diff_3,diff_1])
        # import ipdb;ipdb.set_trace()
        DoG = np.stack([diff_1,diff_2,diff_3,diff_1])

        # DoG = (DoG/DoG.max()).astype(np.float16)
        return(DoG)
    
    def clache(self,bgr,gridsize = 4,clipLimit=2.0):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit,tileGridSize=(gridsize,gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return(cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2GRAY))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.input_dim))
        x = np.zeros((self.batch_size, 4 , *self.input_dim[1:]))
        y = np.zeros((self.batch_size, *self.output_dim))
        imx = np.zeros((self.batch_size, *self.output_dim))
        # import ipdb;ipdb.set_trace()
        # Generate data
        c=[]
        ground_truth=[]
        for i,( idx,selected_id) in enumerate(list_IDs_temp):
            # Store sample
            # import ipdb;ipdb.set_trace()
            # print('\n\n')
            
            data_sample = self._fix_size(idx,selected_id)
            for ix,data in enumerate(data_sample):
                if data != None:
                    actual_file =self.folder+data['name']+'_'+data['frame']+'.jpg'
                    # print(actual_file)
                    # actual_file = data['file-name']
                   # import ipdb;ipdb.set_trace()
                    image_file = cv2.imread(actual_file)
                    image =cv2.cvtColor(image_file,cv2.COLOR_BGR2GRAY)
                    # image = self.clache(image_file,gridsize=5,clipLimit=10.0)
                    if self.cropped == True:
                        image_re = cv2.resize(image,(self.input_dim[1]*2,self.input_dim[2])[::-1])
                        image_re = image_re[int(.5*self.input_dim[1]):int(1.5*self.input_dim[1]),:]
                    else:
                        image_re = cv2.resize(image,self.input_dim[1:][::-1])
                    X[i][ix] = image_re
                    if ix == np.ceil(self.input_dim[0]/2.0):
                        # import ipdb;ipdb.set_trace()
                        # seg_file =self.folder.replace('Image','Seg')+data['name']+'_'+data['frame']+'_seg'+'.jpg'
                        # seg_image_original = cv2.imread(seg_file,self.n_channels)

                        if self.cropped == True:
                                # center = tuple(np.array(data['targets'][0]['center']).astype(np.uint8))
                            seg_image,center = self._create_seg(data,image.shape,(self.output_dim[0]*2,self.output_dim[1]), offset=self.offset)
                            # ce = data['targets'][0]['center']
                            # center = ((ce[0]/float(image.shape[0])*(self.output_dim[0])),(ce[1]/float(image.shape[1])*(self.output_dim[1])))
                            seg_re = seg_image[int(.5*self.output_dim[0]):int(1.5*self.output_dim[0]),:]

                        else:
                            actual_vehicle = data['targets'][0]['category']
                            # center = tuple(np.array(data['targets'][0]['center']).astype(np.uint8))
                            seg_image,center = self._create_seg(data,image.shape,self.output_dim, offset=self.offset)
                            ground_truth.append([actual_vehicle,center])
                            # ce = data['targets'][0]['center']
                            # center = ((ce[1]/float(image.shape[0])*(self.output_dim[0])),(ce[0]/float(image.shape[1])*(self.output_dim[1])))
                            seg_re = seg_image#cv2.resize(seg_image,self.output_dim[::-1])
                        # import ipdb;ipdb.set_trace()
                        # seg_image_reshape = cv2.resize(seg_image,self.output_dim[::-1])\
                        y[i] = seg_re
                        # c.append(center)

                        c.append(tuple(np.array(center).astype(np.uint8)))

            x[i] = self.get_DoG(X[i])
            imx[i] = np.average(np.reshape(x[i],(4,224,224)),axis=0)

            # import ipdb;ipdb.set_trace()
            # self.print_dog([x[i][0],x[i][1],x[i][2],x[i][3]])


        # import ipdb;ipdb.set_trace()
        if self.viz == True:
            return ((X.reshape(self.batch_size,*self.input_dim[1:],5,1),x.reshape(self.batch_size,*self.input_dim[1:],4,1)), (y.reshape(self.batch_size,*self.output_dim,1),c))
        else:
            # im = np.average(np.reshape(x,(4,16,400,500)),axis=0)
            # for idx,i  in enumerate(imx): cv2.imwrite('./x/%d.jpg'%idx,i)
            y = (y>125).astype(np.float16)
            imx=imx/255
            # import ipdb;ipdb.set_trace()
            return ((X,imx.reshape(self.batch_size,*self.output_dim,1)), (ground_truth,y.reshape(self.batch_size,*self.output_dim,1)))

def get_sample_set(data):
    sample = []
    same_sample =[]
    sample_list = []
    last = data[0]['name']
    sample_list.append(last)

    for idx,i in enumerate(data):
        if i['name'] !=last:
            sample.append(same_sample)
            same_sample=[]
            same_sample.append(i)
            last = i['name']
            sample_list.append(last)
        else:
            same_sample.append(i)
    return(sample,sample_list)

def _get_data_fixed(data):
    data_copy = []
    for i in data:
        if len(i['targets']) == 7:
            i['targets']=[i['targets']]
        if len(i['targets']) == 2:
            target=[]
            for pts in i['targets']:
                ul = np.array(pts['ul']) 
                br = np.array(pts['ul'])+2*(np.array(pts['center'])-np.array(pts['ul']))
                target.append((ul,br))
            target = np.array(target)
        
            ul = np.array([min(target[0][0][0],target[1][0][0]),min(target[0][0][1],target[1][0][1])])
            br = np.array([max(target[0][1][0],target[1][1][0]),max(target[0][1][1],target[1][1][1])])
            center = np.average(np.array([ul,br]),axis = 0)
            area = (br-ul)[0]*(br-ul)[1]
            
            data = {'category':'D20_MTLB','id':i['targets'][0]['id'],'inst_id':i['targets'][0]['inst_id'],
            'contrast':(i['targets'][0]['contrast']+i['targets'][1]['contrast'])/2.0,'ul':ul,
            'center':center,'bbox_area':area} 
            i['targets']=[data]

        data_copy.append(i)
    return(data_copy)

def get_data_fixed(data):
    data_copy = []
    for i in data:
        if len(i['targets']) == 7:
            ul = np.array(pts['ul']) 
            c = np.array(pts['center']) 
            if ul[0]-50 >= c[0] or ul[1]-50 >= c[1]:
                continue
            else:
                i['targets']=[i['targets']]
        if len(i['targets']) == 2:
            target=[]
            for pts in i['targets']:
                ul = np.array(pts['ul']) 
                br = np.array(pts['ul'])+2*(np.array(pts['center'])-np.array(pts['ul']))
                c = np.array(pts['center']) 
                if ul[0]-50 >= c[0] or ul[1]-50 >= c[1]:
                    continue
                else:
                    target.append((ul,br))

            target = np.array(target)
            
            if len(target) == 2:
                ul = np.array([min(target[0][0][0],target[1][0][0]),min(target[0][0][1],target[1][0][1])])
                br = np.array([max(target[0][1][0],target[1][1][0]),max(target[0][1][1],target[1][1][1])])
            elif len(target) == 1:
                import ipdb;ipdb.set_trace()
                ul,br = target[0][0],target[0][1]
            else:
                continue
            center = np.average(np.array([ul,br]),axis = 0)
            area = (br-ul)[0]*(br-ul)[1]
            
            data = {'category':'D20_MTLB','id':i['targets'][0]['id'],'inst_id':i['targets'][0]['inst_id'],
            'contrast':(i['targets'][0]['contrast']+i['targets'][1]['contrast'])/2.0,'ul':ul,
            'center':center,'bbox_area':area} 
            i['targets']=[data]

        data_copy.append(i)
    return(data_copy)


def print_data(data,file_list):
    folder = '/'.join(file_list.split('/')[:-1])+'/images/'
    for k in data:
        for i in k:
            # import ipdb;ipdb.set_trace()
            im_file = folder+i['name']+'_'+i['frame']+'.jpg'
            ce = tuple(np.array(i['targets'][0]['center']).astype(np.uint8))
            x= cv2.imread(im_file)
            cv2.circle(x, tuple(ce), radius=2,color = [0,0,255], thickness=4)
            cv2.imwrite('./x/'+i['name']+'_'+i['frame']+'.jpg',x)

if __name__ == '__main__':

    file_list = glob('../atr/scaled*/t*.p')
    pickle_data = [p.load(open(i, 'rb')) for i in file_list]

    train_sample_set = get_sample_set(get_data_fixed(pickle_data[0]))[0]
    
    # print_data([pickle_data[0]],file_list)
    # import ipdb;ipdb.set_trace()

    # train_gen =DataGenerator(train_sample_set, input_dim=(5,200,224), output_dim=(200, 224),viz=True,cropped =True,folder='../atr/scaled2224_1to2/images/',offset=0)

    # for xx in train_gen:
    #     # import ipdb;ipdb.set_trace()
    #     for idx,(X,x,y) in enumerate(zip(xx[0][0],xx[0][1],xx[1])):
    #         # import ipdb;ipdb.set_trace()
    #         #for idy,xx in enumerate(x):

    #         cv2.imwrite('x/%d_actual.jpg'%idx,np.reshape(X,(5, 200, 224))[2])
    #         for idy,i in enumerate((np.reshape(x,( 4,200, 224))*255).astype(np.uint8)):
    #             cv2.imwrite('x/%d-%d_diff.jpg'%(idx,idy),i)
    #         cv2.imwrite('x/%d_seg.jpg'%idx,y)
    #     break

    # train_gen =DataGenerator(train_sample_set, input_dim=(5,400,500), output_dim=(400, 500),viz=False,cropped =False,folder='./atr/scaled2500_1to2/images/',offset=0)
    train_gen =DataGenerator(train_sample_set, input_dim=(5,224,224), output_dim=(224, 224),viz=False,cropped =False,folder='./atr/scaled2500_1to2/images/',offset=0)
    import pandas as pd

    c=0
    col=['filename','count','locations']
    csv = pd.DataFrame(columns=col)
    for batch in train_gen:
        # import ipdb;ipdb.set_trace()
        for idx,(x,xx,y,yy) in enumerate(zip(batch[0][0],batch[0][1],batch[1][0],batch[1][1])):
            # import ipdb;ipdb.set_trace()
            #for idy,xx in enumerate(x):
            import ipdb;ipdb.set_trace()
