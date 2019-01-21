VERBOSE = 1


class __LINE__(object):
    def __repr__(self):
        import sys
        try:
            raise Exception
        except:
            return str(sys.exc_info()[2].tb_frame.f_back.f_lineno)


__LINE__ = __LINE__()


class INFO(object):
    def __init__(self, msg):
        self._msg = msg
        if VERBOSE:
            self.Print()
    
    def Print(self):
        print("[INFO] {}".format(self._msg))


class Warning(INFO):
    def Print(self):
        print("[Warning] {}".format(self._msg))


class Error(INFO):
    def Print(self):
        print("[Error] {}".format(self._msg))
