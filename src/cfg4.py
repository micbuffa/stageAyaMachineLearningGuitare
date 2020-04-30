# Confuguration pour les modeles
import os

class config:
    def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft = nfft
        self.rate=rate

        self.step=int(rate/10)
        self.model_path = os.path.join('models4', mode + '.model')
        self.p_path = os.path.join('pickles4', mode + '.p')