
class MetaData:
    def __init__(self, dataset: str = None, role: str = None, session: str = None):
        self.dataset = dataset
        self.role = role
        self.session = session

    def expand(self, obj_instance):
        new_type = type('Meta', (self.__class__, obj_instance.__class__) , {})
        self.__class__ = new_type
        self.__dict__.update(obj_instance.__dict__)



if __name__ == '__main__':

    class a_meta( ):
        def __init__(self):
            self.a_ = 0

    class b_meta():
        def __init__(self, b = 1):
            self.b_ = b


    meta = MetaData()
    meta.expand(b_meta(5))
    meta.expand(a_meta())
    meta.expand(b_meta(10))