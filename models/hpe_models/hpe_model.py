class HPE_Model():
    def __init__(self):
        self.model_type = 'default'
        self.KEYPOINT_DICT = None
        self.CONNECTIONS = None
        self.conf = 0.4 # default confidence
    
    def predict(self):
        return None

    def get_model_type(self):
        return self.model_type
    
    def get_kpd(self):
        return self.KEYPOINT_DICT
    
    def get_connections(self):
        return self.CONNECTIONS
    
    def get_confidence(self):
        return self.conf
    
    def get_kp_dict(self):
        return self.KEYPOINT_DICT