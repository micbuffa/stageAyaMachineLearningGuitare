# Confuguration for models
import os

class config:
    def __init__(self,mode='conv',nfilt=50,nfeat=25,nfft=512,rate=16000,winlen=0.032,winstep=0.02):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft = nfft
        self.rate=rate
        self.winlen =winlen
        self.winstep =winstep

        self.step=10112
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        self.samples_path = os.path.join('samples', 'samples' + '.smp')
        
        # self.weight_path = os.path.join('rn-1', 'rn-1'+ '.poid')
        # self.kfold_path = os.path.join('kfold', 'kfold'+ '.kf')
       
